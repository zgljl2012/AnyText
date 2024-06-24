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


def arr2tensor(arr, bs, use_fp16=False):
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

def check_overlap_polygon(rect_pts1, rect_pts2):
    poly1 = cv2.convexHull(rect_pts1)
    poly2 = cv2.convexHull(rect_pts2)
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    if rect1[0] + rect1[2] >= rect2[0] and rect2[0] + rect2[2] >= rect1[0] and rect1[1] + rect1[3] >= rect2[1] and rect2[1] + rect2[3] >= rect1[1]:
        return True
    return False

def generate_rectangles(w, h, n, max_trys=200):
    img = np.zeros((h, w, 1), dtype=np.uint8)
    rectangles = []
    attempts = 0
    n_pass = 0
    low_edge = int(max(w, h)*0.3 if n <= 3 else max(w, h)*0.2)  # ~150, ~100
    while attempts < max_trys:
        rect_w = min(np.random.randint(max((w*0.5)//n, low_edge), w), int(w*0.8))
        ratio = np.random.uniform(4, 10)
        rect_h = max(low_edge, int(rect_w/ratio))
        rect_h = min(rect_h, int(h*0.8))
        # gen rotate angle
        rotation_angle = 0
        rand_value = np.random.rand()
        if rand_value < 0.7:
            pass
        elif rand_value < 0.8:
            rotation_angle = np.random.randint(0, 40)
        elif rand_value < 0.9:
            rotation_angle = np.random.randint(140, 180)
        else:
            rotation_angle = np.random.randint(85, 95)
        # rand position
        x = np.random.randint(0, w - rect_w)
        y = np.random.randint(0, h - rect_h)
        # get vertex
        rect_pts = cv2.boxPoints(((rect_w/2, rect_h/2), (rect_w, rect_h), rotation_angle))
        rect_pts = np.int32(rect_pts)
        # move
        rect_pts += (x, y)
        # check boarder
        if np.any(rect_pts < 0) or np.any(rect_pts[:, 0] >= w) or np.any(rect_pts[:, 1] >= h):
            attempts += 1
            continue
        # check overlap
        if any(check_overlap_polygon(rect_pts, rp) for rp in rectangles):
            attempts += 1
            continue
        n_pass += 1
        cv2.fillPoly(img, [rect_pts], 255)
        rectangles.append(rect_pts)
        if n_pass == n:
            break
    print("attempts:", attempts)
    if len(rectangles) != n:
        raise Exception(f'Failed in auto generate positions after {attempts} attempts, try again!')
    return img
