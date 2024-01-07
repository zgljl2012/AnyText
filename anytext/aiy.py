from modelscope.pipelines import pipeline
from anytext.util import save_images
from PIL import Image
from typing import List
import sys
import os

# 添加本路径
DIR = os.path.dirname(__file__)
sys.path.append(DIR)

class AiyAnyText:
    cfg_path: str
    font_path: str

    def __init__(self, font_path=None):
        self.font_path = font_path
        if self.font_path is None:
            self.font_path = os.path.join(DIR, 'font/Arial_Unicode.ttf')
        self.cfg_path = os.path.join(DIR, 'models_yaml/anytext_sd15.yaml')
        self._init_pipe()

    def _init_pipe(self):
        self.pipe = pipeline('my-anytext-task', cfg_path=self.cfg_path,
                                 font_path=self.font_path,
                                 model='damo/cv_anytext_text_generation_editing',
                                 model_revision='v1.1.1',
                                 use_fp16=False)
    
    def text_generation(self, prompt: str, draw_pos_path=str, n_steps: int=20, seed: int=1, show_debug=True, image_count=1) -> List[Image.Image]:
        params = {
            "show_debug": show_debug,
            "image_count": image_count,
            "ddim_steps": n_steps,
            "image_width": 512,
            "image_height": 512
        }

        # 1. text generation
        mode = 'text-generation'
        input_data = {
            "prompt": prompt,
            "seed": seed,
            "draw_pos": draw_pos_path
        }
        results, rtn_code, rtn_warning, debug_info = self.pipe(input_data, mode=mode, **params)
        if rtn_warning:
            print(rtn_warning)
        if rtn_code >= 0:
            imgs = [Image.fromarray(i) for i in results]
            return imgs
        return []

def test():
    font_path = os.path.join(DIR, 'font/Arial_Unicode.ttf')
    cfg_path = os.path.join(DIR, 'models_yaml/anytext_sd15.yaml')
    pipe = pipeline('my-anytext-task', cfg_path=cfg_path, font_path=font_path, model='damo/cv_anytext_text_generation_editing', model_revision='v1.1.1', use_fp16=False)
    img_save_folder = "SaveImages"
    params = {
        "show_debug": True,
        "image_count": 2,
        "ddim_steps": 20,
    }

    # 1. text generation
    mode = 'text-generation'
    input_data = {
        "prompt": 'photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream',
        "seed": 66273235,
        "draw_pos": 'example_images/gen9.png'
    }
    results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
    if rtn_code >= 0:
        save_images(results, img_save_folder)
        print(f'Done, result images are saved in: {img_save_folder}')
    if rtn_warning:
        print(rtn_warning)
    # 2. text editing
    mode = 'text-editing'
    input_data = {
        "prompt": 'A cake with colorful characters that reads "EVERYDAY"',
        "seed": 8943410,
        "draw_pos": 'example_images/edit7.png',
        "ori_image": 'example_images/ref7.jpg'
    }
    results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
    if rtn_code >= 0:
        save_images(results, img_save_folder)
        print(f'Done, result images are saved in: {img_save_folder}')
    if rtn_warning:
        print(rtn_warning)
