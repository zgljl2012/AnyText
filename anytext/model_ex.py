
from anytext.cldm.model import create_model
import torch
import os
import sys

# 添加本路径
DIR = os.path.dirname(__file__)
sys.path.append(DIR)

model_path = 'E://.modelscope/damo/cv_anytext_text_generation_editing'

def extract_control_model():
    """ 分离出 control model 的参数 """
    cfg_path = "anytext/models_yaml/anytext_sd15.yaml"
    ckpt_path = os.path.join(model_path, "anytext_v1.1.ckpt")
    clip_path = os.path.join(model_path, "clip-vit-large-patch14")
    model = create_model(
        cfg_path, cond_stage_path=clip_path, use_fp16=False
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')), strict=False)
    
    control_model = model.control_model
    torch.save(control_model.state_dict(), os.path.join(model_path, f'anytext_control_model_v1.1.ckpt'))

def extract_unet_model():
    cfg_path = "anytext/models_yaml/anytext_sd15.yaml"
    ckpt_path = os.path.join(model_path, "anytext_v1.1.ckpt")
    clip_path = os.path.join(model_path, "clip-vit-large-patch14")
    model = create_model(
        cfg_path, cond_stage_path=clip_path, use_fp16=False
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')), strict=False)
    
    unet = model.model.diffusion_model
    torch.save(unet.state_dict(), os.path.join(model_path, f'anytext_unet_v1.1.ckpt'))

def load_control_model():
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
    ckpt_path = os.path.join(model_path, f'anytext_control_model_v1.1.ckpt')
    control_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cuda')), strict=False)

if __name__ == '__main__':
    # extract_control_model()
    # load_control_model()
    extract_unet_model()
