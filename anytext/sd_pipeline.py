import inspect
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import LCMScheduler
from diffusers.utils.torch_utils import randn_tensor
import random
import torch
import time
from tqdm import tqdm

def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


@torch.no_grad()
def pipeline():
    start_at = time.time()
    prompt = 'a dog'
    n_steps = 15
    model_path = 'E://civital/dreamshaper_8LCM'
    dtype = torch.float16
    device = torch.device("cuda")
    height = 512
    width = 512
    seed = random.randint(1, 1000000000)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # tokenization
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    # clip
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype).cuda()
    
    prompt_embeds_output: BaseModelOutputWithPooling = text_encoder(text_input_ids.to(device))
    prompt_embeds = prompt_embeds_output[0]
    # unet
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).cuda()
    num_channels_latents = unet.config.in_channels
    # vae
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).cuda()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    # scheduler
    scheduler: LCMScheduler = LCMScheduler.from_pretrained(model_path, subfolder='scheduler')
    scheduler.set_timesteps(n_steps, device=device)
    timesteps = scheduler.timesteps
    
    eta = 0
    extra_step_kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)
    
    shape = (1, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    
    with tqdm(total=n_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = latents
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs={},
                return_dict=False,
            )[0]
            latents = scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]
            # update progress
            progress_bar.update()
    
    # output
    images = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
    image = image_processor.postprocess(images, output_type='pil', do_denormalize=[True] * images.shape[0])[0]
    file = 'test-output.png'
    image.save(file)
    print(f'Saved {file}')
    # end
    print('Time:', (time.time() - start_at), 's')

if __name__ == '__main__':
    pipeline()
