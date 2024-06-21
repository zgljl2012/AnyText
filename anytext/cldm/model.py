import os
import torch

from omegaconf import OmegaConf
from anytext.ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path, cond_stage_path=None, use_fp16=False):
    config = OmegaConf.load(config_path)
    if cond_stage_path:
        config.model.params.cond_stage_config.params.version = cond_stage_path  # use pre-downloaded ckpts, in case blocked
    if use_fp16:
        config.model.params.use_fp16 = True
        config.model.params.control_stage_config.params.use_fp16 = True
        config.model.params.unet_config.params.use_fp16 = True
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def create_model1():
    # from  cldm.embedding_manager import EmbeddingManager
    embedding_manager_config = {
        "valid": True,  # v6
        "emb_type": 'conv',  # ocr, vit, conv
        "glyph_channels": 1,
        "position_channels": 1,
        "add_pos": False,
        "placeholder_string": '*'
    }
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
        legacy=False
    )
    from cldm.cldm import ControlledUnetModel
    unet_config = ControlledUnetModel(
        image_size=32, # unused
        in_channels=4,
        out_channels=4,
        model_channels=320,
        attention_resolutions=[ 4, 2, 1 ],
        num_res_blocks=2,
        channel_mult=[ 1, 2, 4, 4 ],
        num_heads=8,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=768,
        use_checkpoint=True,
        legacy=False
    )
    from ldm.models.autoencoder import AutoencoderKL
    first_stage_config = AutoencoderKL(
        embed_dim=4,
        monitor='val/rec_loss',
        ddconfig= {
          "double_z": True,
          "z_channels": 4,
          "resolution": 256,
          "in_channels": 3,
          "out_ch": 3,
          "ch": 128,
          "ch_mult": [1,2,4,4],
          "num_res_blocks": 2,
          "attn_resolutions": [],
          "dropout": 0.0
        },
        lossconfig={
          "target": "torch.nn.Identity"
        }
    )
    from ldm.modules.encoders.modules import FrozenCLIPEmbedderT3
    cond_stage_config = FrozenCLIPEmbedderT3(
        version='./models/clip-vit-large-patch14',
        use_vision=False  # v6
    )
    from cldm.cldm1 import ControlLDM1
    control_ldm = ControlLDM1(
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps_cond=1,
        log_every_t=200,
        timesteps=1000,
        first_stage_key="img",
        cond_stage_key="caption",
        control_key="hint",
        glyph_key="glyphs",
        position_key="positions",
        image_size=64,
        channels=4,
        cond_stage_trainable=True , # need be true when embedding_manager is valid
        conditioning_key='crossattn',
        monitor='val/loss_simple_ema',
        scale_factor=0.18215,
        use_ema=False,
        only_mid_control=False,
        loss_alpha=0,  # perceptual loss, 0.003
        loss_beta=0 , # ctc loss
        latin_weight=1.0 , # latin text line may need smaller weigth
        with_step_weight=True,
        use_vae_upsample=True,
        embedding_manager_config=embedding_manager_config,
        control_model=control_model,
        unet_config=unet_config,
        first_stage_config=first_stage_config,
        cond_stage_config=cond_stage_config,
    )
    return control_ldm
