from anytext.util import save_images
from PIL import Image
from typing import List, Optional
import sys
import os
import torch

# 添加本路径
DIR = os.path.dirname(__file__)
sys.path.append(DIR)


from anytext.aiy_text.ms_wrapper import AnyTextModel

def pipeline(model: str = None,
             cfg_path=None,
             font_path=None,
             device: str = 'gpu'):
    pipeline_props = {
        "use_fp16": False
    }
    pipeline_props['model'] = model
    pipeline_props['device'] = device
    pipeline_props['cfg_path'] = cfg_path
    pipeline_props['font_path'] = font_path
    # return AnyTextPipeline(**pipeline_props)
    pipe_model = AnyTextModel(model_dir=model, **pipeline_props)
    return pipe_model

class AiyAnyText:
    cfg_path: str
    font_path: str

    def __init__(self, font_path=None):
        self.font_path = font_path
        if self.font_path is None:
            self.font_path = os.path.join(DIR, "font/Arial_Unicode.ttf")
        self.cfg_path = os.path.join(DIR, "models_yaml/anytext_sd15.yaml")
        self._init_pipe()

    def _init_pipe(self):
        # 创建 Pipeline
        self.pipe = pipeline(
            cfg_path=self.cfg_path,
            font_path=self.font_path,
            model="E://.modelscope\damo\cv_anytext_text_generation_editing",
        )

    def text_generation(
        self,
        prompt: str,
        draw_pos_path: str,
        ori_image: str=None,
        n_steps: int = 20,
        seed: int = 1,
        show_debug=True,
        image_count=1,
        width=512,
        height=512,
        mode = 'text-generation' # 'text-generation', 'text-editing'
    ) -> List[Image.Image]:
        params = {
            "show_debug": show_debug,
            "image_count": image_count,
            "ddim_steps": n_steps,
            "image_width": width,
            "image_height": height,
        }

        # 1. text generation
        input_data = {"prompt": prompt, "seed": seed, "draw_pos": draw_pos_path, 'ori_image': ori_image}
        self.pipe(
            input_data, mode=mode, **params
        )
        return []
