"""
AnyText: Multilingual Visual Text Generation And Editing
Paper: https://arxiv.org/abs/2311.03054
Code: https://github.com/tyxsspa/AnyText
Copyright (c) Alibaba, Inc. and its affiliates.
"""

import os
from typing import Any, Dict, List, Union, Generator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import random
import re
import numpy as np
import cv2
from PIL import ImageFont
from anytext.t3_dataset import draw_glyph, draw_glyph2
from anytext.util import check_channels, resize_image
from pytorch_lightning import seed_everything
from PIL import Image
import numpy

Input = Union[str, tuple, 'Image.Image', 'numpy.ndarray']

PLACE_HOLDER = "*"
max_chars = 20

from .utils import is_chinese, separate_pos_imgs, arr2tensor, find_polygon, prepare_extra_step_kwargs


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class AnyTextModel(torch.nn.Module):

    def __init__(self, model_dir, **kwargs):
        super().__init__()
        self.model_dir = model_dir
        self.use_fp16 = kwargs.get("use_fp16", True)
        self.use_translator = kwargs.get("use_translator", True)
        self.init_model(**kwargs)

    """
    return:
        result: list of images in numpy.ndarray format
        rst_code: 0: normal -1: error 1:warning
        rst_info: string of error or warning
        debug_info: string for debug, only valid if show_debug=True
    """

    @torch.no_grad()
    def forward(self, input_tensor, **forward_params):
        # get inputs
        seed = input_tensor.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 99999999)
        seed_everything(seed)
        prompt = input_tensor.get("prompt")
        draw_pos = input_tensor.get("draw_pos")
        ori_image = input_tensor.get("ori_image")

        mode = forward_params.get("mode")
        sort_priority = forward_params.get("sort_priority", "↕")
        revise_pos = forward_params.get("revise_pos", False)
        img_count = forward_params.get("image_count", 4)
        ddim_steps = forward_params.get("ddim_steps", 20)
        w = forward_params.get("image_width", 512)
        h = forward_params.get("image_height", 512)
        strength = forward_params.get("strength", 1.0)
        cfg_scale = forward_params.get("cfg_scale", 9.0)
        eta = forward_params.get("eta", 0.0)
        a_prompt = forward_params.get(
            "a_prompt",
            "best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks",
        )
        n_prompt = forward_params.get(
            "n_prompt",
            "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
        )

        prompt, texts = self.modify_prompt(prompt)
        if prompt is None and texts is None:
            return (
                None,
                -1,
                "You have input Chinese prompt but the translator is not loaded!",
                "",
            )
        n_lines = len(texts)
        if mode in ["text-generation", "gen"]:
            edit_image = np.ones((h, w, 3)) * 127.5  # empty mask image
        elif mode in ["text-editing", "edit"]:
            if draw_pos is None or ori_image is None:
                return (
                    None,
                    -1,
                    "Reference image and position image are needed for text editing!",
                    "",
                )
            if isinstance(ori_image, str):
                ori_image = cv2.imread(ori_image)[..., ::-1]
                assert (
                    ori_image is not None
                ), f"Can't read ori_image image from{ori_image}!"
            elif isinstance(ori_image, torch.Tensor):
                ori_image = ori_image.cpu().numpy()
            else:
                assert isinstance(
                    ori_image, np.ndarray
                ), f"Unknown format of ori_image: {type(ori_image)}"
            edit_image = ori_image.clip(1, 255)  # for mask reason
            edit_image = check_channels(edit_image)
            edit_image = resize_image(
                edit_image, max_length=768
            )  # make w h multiple of 64, resize if w or h > max_length
            h, w = edit_image.shape[:2]  # change h, w by input ref_img
        # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
        if draw_pos is None:
            pos_imgs = np.zeros((w, h, 1))
        if isinstance(draw_pos, str):
            draw_pos = cv2.imread(draw_pos)[..., ::-1]
            assert draw_pos is not None, f"Can't read draw_pos image from{draw_pos}!"
            pos_imgs = 255 - draw_pos
        elif isinstance(draw_pos, torch.Tensor):
            pos_imgs = draw_pos.cpu().numpy()
        else:
            assert isinstance(
                draw_pos, np.ndarray
            ), f"Unknown format of draw_pos: {type(draw_pos)}"
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        # seprate pos_imgs
        pos_imgs = separate_pos_imgs(pos_imgs, sort_priority)
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == " ":
                pass  # text-to-image without text
            else:
                return (
                    None,
                    -1,
                    f"Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!",
                    "",
                )
        elif len(pos_imgs) > n_lines:
            str_warning = f"Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt."
        # get pre_pos, poly_list, hint that needed for anytext
        pre_pos = []
        poly_list = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = (
                    input_pos[..., np.newaxis]
                    if len(input_pos.shape) == 2
                    else input_pos
                )
                poly, pos_img = find_polygon(input_pos)
                pre_pos += [pos_img / 255.0]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((h, w, 1))]
                poly_list += [None]
        np_hint = np.sum(pre_pos, axis=0).clip(0, 1)
        # prepare info dict
        info = {}
        info["glyphs"] = []
        info["gly_line"] = []
        info["positions"] = []
        info["n_lines"] = [len(texts)] * img_count
        gly_pos_imgs = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                str_warning = (
                    f'"{text}" length > max_chars: {max_chars}, will be cut off...'
                )
                text = text[:max_chars]
            gly_scale = 2
            if pre_pos[i].mean() != 0:
                gly_line = draw_glyph(self.font, text)
                glyphs = draw_glyph2(
                    self.font,
                    text,
                    poly_list[i],
                    scale=gly_scale,
                    width=w,
                    height=h,
                    add_space=False,
                )
                gly_pos_img = cv2.drawContours(
                    glyphs * 255, [poly_list[i] * gly_scale], 0, (255, 255, 255), 1
                )
                if revise_pos:
                    resize_gly = cv2.resize(
                        glyphs, (pre_pos[i].shape[1], pre_pos[i].shape[0])
                    )
                    new_pos = cv2.morphologyEx(
                        (resize_gly * 255).astype(np.uint8),
                        cv2.MORPH_CLOSE,
                        kernel=np.ones(
                            (resize_gly.shape[0] // 10, resize_gly.shape[1] // 10),
                            dtype=np.uint8,
                        ),
                        iterations=1,
                    )
                    new_pos = (
                        new_pos[..., np.newaxis] if len(new_pos.shape) == 2 else new_pos
                    )
                    contours, _ = cv2.findContours(
                        new_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    if len(contours) != 1:
                        str_warning = f"Fail to revise position {i} to bounding rect, remain position unchanged..."
                    else:
                        rect = cv2.minAreaRect(contours[0])
                        poly = np.int0(cv2.boxPoints(rect))
                        pre_pos[i] = (
                            cv2.drawContours(new_pos, [poly], -1, 255, -1) / 255.0
                        )
                        gly_pos_img = cv2.drawContours(
                            glyphs * 255, [poly * gly_scale], 0, (255, 255, 255), 1
                        )
                gly_pos_imgs += [gly_pos_img]  # for show
            else:
                glyphs = np.zeros((h * gly_scale, w * gly_scale, 1))
                gly_line = np.zeros((80, 512, 1))
                gly_pos_imgs += [
                    np.zeros((h * gly_scale, w * gly_scale, 1))
                ]  # for show
            pos = pre_pos[i]
            info["glyphs"] += [arr2tensor(glyphs, img_count)]
            info["gly_line"] += [arr2tensor(gly_line, img_count)]
            info["positions"] += [arr2tensor(pos, img_count)]
        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0) * (1 - np_hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().cuda()
        if self.use_fp16:
            masked_img = masked_img.half()
        #-----------------------------------
        from diffusers import AutoencoderKL
        model_path = "E://civital/dreamshaper_8LCM"
        # model_path = 'D:\workspace/aiy-server\data\models\huggingface/runwayml\stable-diffusion-v1-5'
        device = torch.device('cuda')
        dtype = torch.float32
        use_fp16 = False
        #--------
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=dtype
        ).cuda()
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        from diffusers.image_processor import VaeImageProcessor
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        encoder_posterior = vae.encode(masked_img[None, ...])[0]
        scale_factor = 0.18215
        masked_x = (scale_factor * encoder_posterior.sample()).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        info["masked_x"] = torch.cat([masked_x for _ in range(img_count)], dim=0)

        hint = arr2tensor(np_hint, img_count)
        #-------------------------------------------------------------
        from anytext.cldm.embedding_manager import EmbeddingManager
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
        from easydict import EasyDict as edict

        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        
        text_inputs = tokenizer(
            prompt + " , " + a_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        #  tokenizer 结果
        text_input_ids = text_inputs.input_ids

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
        ).cuda()
        # recog
        args = edict()
        args.rec_image_shape = "3, 48, 320"
        args.rec_batch_num = 6
        args.rec_char_dict_path = "anytext/ocr_recog/ppocr_keys_v1.txt"
        args.use_fp16 = use_fp16
        from anytext.cldm.recognizer import TextRecognizer, create_predictor
        text_predictor = create_predictor(use_fp16=use_fp16).eval()
        text_predictor = text_predictor.cuda()
        cn_recognizer = TextRecognizer(args, text_predictor)
        embedding_manager.recog = cn_recognizer
        embedding_manager.encode_text(info)
        #-------------------------------
        from .aiy_clip_text_model import AiyCLIPTextModel
        text_encoder: AiyCLIPTextModel = AiyCLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        ).cuda()
        prompt_embeds = text_encoder(
            text_input_ids.to(device), embedding_manager=embedding_manager
        )
        #---------------------------------------------------------------
        shape = (4, h // 8, w // 8)
        control_scales = [strength] * 13
        
        #------------------------------------------
        batch_size = img_count
        # unconditional_guidance_scale = cfg_scale
        # unconditional_conditioning = un_cond
        
        #------------ control model-----
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
            use_fp16=use_fp16
        )
        import os
        model_path1 = 'E://.modelscope/damo/cv_anytext_text_generation_editing'
        ckpt_path = os.path.join(model_path1, f'anytext_control_model_v1.1.ckpt')
        control_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')), strict=False)
        control_model = control_model.to(device=device, dtype=dtype)
        #---------------------------------
        
        from diffusers.schedulers import DDIMScheduler, LCMScheduler
        scheduler: LCMScheduler = LCMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        scheduler.set_timesteps(ddim_steps, device=device)
        timesteps = scheduler.timesteps
        
        seed = random.randint(1, 1000000000)
        generator = torch.Generator(device=device).manual_seed(seed)
        extra_step_kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)
        
        #------------------------------------------------------
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", torch_dtype=dtype
        ).cuda()
        #------------------------------------------------------

        # sampling
        C, H, W = shape
        shape = (batch_size, C, H, W)
        print(f"Data shape for DDIM sampling is {shape}, eta {eta}")
        
        #------
        b = shape[0]

        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        img = torch.randn(shape, device=device)
        from tqdm import tqdm

        with tqdm(total=total_steps) as progress_bar:
            for i, step in enumerate(timesteps):
                # index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)

                #-----------------------------------------------------------------------
                x = img
                t = ts
                
                # 先完全忽略负面提示词
                # assert unconditional_conditioning is None
                #------------------------------------------------------------------
                # assert isinstance(cond, dict)
                _cond = prompt_embeds
                _hint = hint
                if use_fp16:
                    x = x.half()
                control = control_model(x=x, timesteps=ts, context=_cond, hint=_hint, text_info=info)
                control = [c * scale for c, scale in zip(control, control_scales)]
                
                mid_block_additional_residual = control.pop()
                down_block_additional_residuals = control
                
                noise_pred = unet(
                    x,
                    t,
                    encoder_hidden_states=_cond,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs={},
                    return_dict=False,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_block_additional_residuals=down_block_additional_residuals,
                )[0]
                e_t = noise_pred
                
                x_prev = scheduler.step(
                    e_t,
                    step,
                    x,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                
                #-------------------------------------------------------------------------
                img = x_prev
                progress_bar.update()

        samples = img

        #--------------------------------------
        images = vae.decode(
            samples / vae.config.scaling_factor, return_dict=False, generator=generator
        )[0]
        images = image_processor.postprocess(
            images, output_type="pil", do_denormalize=[True] * images.shape[0]
        )
        for i, image in enumerate(images):
            file = f"test-output-{i}.png"
            image.save(file)
            print(f"Saved {file}")


    def init_model(self, **kwargs):
        font_path = kwargs.get("font_path", "font/Arial_Unicode.ttf")
        self.font = ImageFont.truetype(font_path, size=60)

    def modify_prompt(self, prompt):
        prompt = prompt.replace("“", '"')
        prompt = prompt.replace("”", '"')
        p = '"(.*?)"'
        strs = re.findall(p, prompt)
        if len(strs) == 0:
            strs = [" "]
        else:
            for s in strs:
                prompt = prompt.replace(f'"{s}"', f" {PLACE_HOLDER} ", 1)
        if is_chinese(prompt):
            if self.trans_pipe is None:
                return None, None
            old_prompt = prompt
            prompt = self.trans_pipe(input=prompt + " .")["translation"][:-1]
            print(f"Translate: {old_prompt} --> {prompt}")
        return prompt, strs
