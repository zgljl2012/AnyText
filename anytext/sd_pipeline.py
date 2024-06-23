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
import re
import numpy as np
import cv2
from easydict import EasyDict as edict

from anytext.cldm.recognizer import TextRecognizer, create_predictor

PLACE_HOLDER = "*"

from anytext.bert_tokenizer import BasicTokenizer

checker = BasicTokenizer()


def is_chinese(text):
    text = checker._clean_text(text)
    for char in text:
        cp = ord(char)
        if checker._is_chinese_char(cp):
            return True
    return False


def modify_prompt(prompt):
    prompt = prompt.replace("“", '"')
    prompt = prompt.replace("”", '"')
    # 提取所有引号内的文本
    p = '"(.*?)"'
    strs = re.findall(p, prompt)
    if len(strs) == 0:
        strs = [" "]
    else:
        for s in strs:
            prompt = prompt.replace(f'"{s}"', f" {PLACE_HOLDER} ", 1)
    if is_chinese(prompt):
        # TODO 支持中文
        raise NotImplemented
    return prompt, strs


def separate_pos_imgs(img, sort_priority, gap=102):
    num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(img)
    components = []
    for label in range(1, num_labels):
        component = np.zeros_like(img)
        component[labels == label] = 255
        components.append((component, centroids[label]))
    if sort_priority == "↕":
        fir, sec = 1, 0  # top-down first
    elif sort_priority == "↔":
        fir, sec = 0, 1  # left-right first
    components.sort(key=lambda c: (c[1][fir] // gap, c[1][sec] // gap))
    sorted_components = [c[0] for c in components]
    return sorted_components


def find_polygon(image, min_rect=False):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea)  # get contour with max area
    if min_rect:
        # get minimum enclosing rectangle
        rect = cv2.minAreaRect(max_contour)
        poly = np.int0(cv2.boxPoints(rect))
    else:
        # get approximate polygon
        epsilon = 0.01 * cv2.arcLength(max_contour, True)
        poly = cv2.approxPolyDP(max_contour, epsilon, True)
        n, _, xy = poly.shape
        poly = poly.reshape(n, xy)
    cv2.drawContours(image, [poly], -1, 255, -1)
    return poly, image


def arr2tensor(arr, bs, use_fp16):
    arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().cuda()
    if use_fp16:
        _arr = _arr.half()
    _arr = torch.stack([_arr for _ in range(bs)], dim=0)
    return _arr


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
    accepts_generator = "generator" in set(
        inspect.signature(scheduler.step).parameters.keys()
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

@torch.no_grad()
def anytext1():
    import cv2
    from anytext.t3_dataset import draw_glyph, draw_glyph2
    from PIL import ImageFont

    model_path = "E://civital/dreamshaper_8LCM"
    dtype = torch.float16
    device = torch.device("cuda")
    height = 512
    width = 512
    prompt = 'photo of caramel macchiato coffee on the table, top-down perspective, with "Any" written on it using cream'
    use_fp16 = True
    ### AnyText: text embedding module
    font_path = "anytext/font/Arial_Unicode.ttf"
    font = ImageFont.truetype(font_path, size=60)
    ## glyph render
    # 提取提示词和文本
    prompt, texts = modify_prompt(prompt)
    print(prompt, "---", texts)

    # 现在支持的是 "text-generation", "gen" mode
    # TODO 支持 "text-editing", "edit" mode
    edit_image = np.ones((height, width, 3)) * 127.5  # empty mask image

    pos_imgs = np.zeros((width, height, 1))

    pos_imgs = pos_imgs[..., 0:1]
    pos_imgs = cv2.convertScaleAbs(pos_imgs)
    _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
    # seprate pos_imgs
    sort_priority = "↕"
    pos_imgs = separate_pos_imgs(pos_imgs, sort_priority)

    pos_imgs = [np.zeros((height, width, 1))]
    # get pre_pos, poly_list, hint that needed for anytext
    pre_pos = []
    poly_list = []
    for input_pos in pos_imgs:
        if input_pos.mean() != 0:
            input_pos = (
                input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
            )
            poly, pos_img = find_polygon(input_pos)
            pre_pos += [pos_img / 255.0]
            poly_list += [poly]
        else:
            pre_pos += [np.zeros((height, width, 1))]
            poly_list += [None]
    np_hint = np.sum(pre_pos, axis=0).clip(0, 1)
    # prepare info dict
    info = {}
    info["glyphs"] = []
    info["gly_line"] = []
    info["positions"] = []
    img_count = 1
    max_chars = 20
    info["n_lines"] = [len(texts)] * img_count
    gly_pos_imgs = []
    for i in range(len(texts)):
        text = texts[i]
        if len(text) > max_chars:
            str_warning = (
                f'"{text}" length > max_chars: {max_chars}, will be cut off...'
            )
            print(str_warning)
            text = text[:max_chars]
        gly_scale = 2
        if pre_pos[i].mean() != 0:
            gly_line = draw_glyph(font, text)
            glyphs = draw_glyph2(
                font,
                text,
                poly_list[i],
                scale=gly_scale,
                width=width,
                height=height,
                add_space=False,
            )
            gly_pos_img = cv2.drawContours(
                glyphs * 255, [poly_list[i] * gly_scale], 0, (255, 255, 255), 1
            )
            gly_pos_imgs += [gly_pos_img]  # for show
        else:
            glyphs = np.zeros((height * gly_scale, width * gly_scale, 1))
            gly_line = np.zeros((80, 512, 1))
            gly_pos_imgs += [
                np.zeros((height * gly_scale, width * gly_scale, 1))
            ]  # for show
        pos = pre_pos[i]
        info["glyphs"] += [arr2tensor(glyphs, img_count, use_fp16)]
        info["gly_line"] += [arr2tensor(gly_line, img_count, use_fp16)]
        info["positions"] += [arr2tensor(pos, img_count, use_fp16)]

    # get masked_x
    masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0) * (1 - np_hint)
    masked_img = np.transpose(masked_img, (2, 0, 1))
    masked_img = torch.from_numpy(masked_img.copy()).float().cuda()
    if use_fp16:
        masked_img = masked_img.half()
    # 实际调用的是 first_stage_model.encode() -> AutoencoderKL.encode()
    # 对应到 diffusers 中，就是调用的 Vae 模型
    # encoder_posterior = self.model.encode_first_stage(masked_img[None, ...])
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    ).cuda()
    encoder_posterior = vae.encode(masked_img[None, ...])[0].sample()
    masked_x = encoder_posterior.detach()
    if use_fp16:
        masked_x = masked_x.half()
    info["masked_x"] = torch.cat([masked_x for _ in range(img_count)], dim=0).half()

    hint = arr2tensor(np_hint, img_count, use_fp16)
    a_prompt = "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks"
    from anytext.cldm.embedding_manager import EmbeddingManager

    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")

    class A:
        def __init__(self):
            self.tokenizer = tokenizer

    embedding_manager = EmbeddingManager(
        embedder=A(),
        emb_type="ocr",  # ocr, vit, conv
        glyph_channels=1,
        position_channels=1,
        add_pos=False,
        placeholder_string="*",
        use_fp16=use_fp16,
    )
    # recog
    args = edict()
    args.rec_image_shape = "3, 48, 320"
    args.rec_batch_num = 6
    args.rec_char_dict_path = "anytext/ocr_recog/ppocr_keys_v1.txt"
    args.use_fp16 = use_fp16
    text_predictor = create_predictor(use_fp16=use_fp16).eval()
    cn_recognizer = TextRecognizer(args, text_predictor)
    embedding_manager.recog = cn_recognizer
    embedding_manager.encode_text(info)
    from anytext.ldm.modules.encoders.modules import FrozenCLIPEmbedderT3

    cond_stage_model = (
        FrozenCLIPEmbedderT3(
            version="E://.modelscope/damo/cv_anytext_text_generation_editing/clip-vit-large-patch14"
        )
        .cuda()
        .to(dtype=torch.float16)
    )
    cond_txt = cond_stage_model.encode(
        [prompt + " , " + a_prompt] * img_count, embedding_manager=embedding_manager
    )
    shape = (4, height // 8, width // 8)
    strength = 1.0
    control_scales = [strength] * 13
    
    # control model
    from cldm.cldm import ControlNet
    control_model = ControlNet(
        image_size=32, # unused
        in_channels=4,
        model_channels=320,
        glyph_channels=1,
        position_channels=1,
        attention_resolutions=[ 4, 2, 1 ],
        num_res_blocks=2,
        channel_mult=[ 1, 2, 4, 4 ],
        num_heads=8,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=768,
        use_checkpoint=True,
        legacy=False,
        use_fp16=True
    )
    import os
    model_path1 = 'E://.modelscope/damo/cv_anytext_text_generation_editing'
    ckpt_path = os.path.join(model_path1, f'anytext_control_model_v1.1.ckpt')
    control_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')), strict=False)
    control_model = control_model.to(device=device, dtype=dtype)
    # scheduler
    n_steps = 20
    
    start_at = time.time()
    seed = random.randint(1, 1000000000)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    prompt_embeds = cond_txt
    
    # unet
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", torch_dtype=dtype
    ).cuda()
    num_channels_latents = unet.config.in_channels
    # vae
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    ).cuda()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    # scheduler
    scheduler: LCMScheduler = LCMScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )
    scheduler.set_timesteps(n_steps, device=device)
    timesteps = scheduler.timesteps

    eta = 0
    extra_step_kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)

    shape = (
        1,
        num_channels_latents,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    with tqdm(total=n_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = latents
            
            # controlnet
            ts = torch.full((shape[0],), t, device=device, dtype=torch.long)
            control = control_model(x=latent_model_input, timesteps=ts, context=prompt_embeds, hint=hint, text_info=info)
            control = [c * scale for c, scale in zip(control, control_scales)]
            
            mid_block_additional_residual = control.pop()
            down_block_additional_residuals = control
            
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs={},
                return_dict=False,
                mid_block_additional_residual=mid_block_additional_residual,
                down_block_additional_residuals=down_block_additional_residuals,
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
    images = vae.decode(
        latents / vae.config.scaling_factor, return_dict=False, generator=generator
    )[0]
    image = image_processor.postprocess(
        images, output_type="pil", do_denormalize=[True] * images.shape[0]
    )[0]
    file = "test-output.png"
    image.save(file)
    print(f"Saved {file}")
    # end
    print("Time:", (time.time() - start_at), "s")


@torch.no_grad()
def pipeline():
    start_at = time.time()
    prompt = "a dog"
    n_steps = 15
    model_path = "E://civital/dreamshaper_8LCM"
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
    #  tokenizer 结果
    text_input_ids = text_inputs.input_ids

    # clip
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype
    ).cuda()

    prompt_embeds_output: BaseModelOutputWithPooling = text_encoder(
        text_input_ids.to(device)
    )
    prompt_embeds = prompt_embeds_output[0]
    # unet
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", torch_dtype=dtype
    ).cuda()
    num_channels_latents = unet.config.in_channels
    # vae
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    ).cuda()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    # scheduler
    scheduler: LCMScheduler = LCMScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )
    scheduler.set_timesteps(n_steps, device=device)
    timesteps = scheduler.timesteps

    eta = 0
    extra_step_kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)

    shape = (
        1,
        num_channels_latents,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
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
    images = vae.decode(
        latents / vae.config.scaling_factor, return_dict=False, generator=generator
    )[0]
    image = image_processor.postprocess(
        images, output_type="pil", do_denormalize=[True] * images.shape[0]
    )[0]
    file = "test-output.png"
    image.save(file)
    print(f"Saved {file}")
    # end
    print("Time:", (time.time() - start_at), "s")


if __name__ == "__main__":
    # pipeline()
    anytext1()
