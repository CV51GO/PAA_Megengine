import numpy as np
import time
import collections
import megengine
import megengine.functional as F
import megengine.hub as hub
from paa import get_megengine_hardnet_model
import random
import sys
sys.path.append('./official_PAA')
from paa_core.config import cfg
from paa_core.modeling.detector import build_detection_model
from paa_core.utils.checkpoint import DetectronCheckpointer
from pathlib import Path

import torch
from PIL import Image
import os

class ImageList(object):
    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

def to_image_list_torch(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    if size_divisible > 0:
        import math

        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(tensors),) + max_size
    batched_imgs = tensors[0].new(*batch_shape).zero_()

    for img, pad_img in zip(tensors, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img

    image_sizes = [im.shape[-2:] for im in tensors]

    return ImageList(batched_imgs, image_sizes)

def to_image_list_megengine(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))
    if size_divisible > 0:
        import math

        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(tensors),) + max_size
    batched_imgs = F.zeros(batch_shape, dtype=tensors[0].dtype)
    batched_imgs[0,: tensors[0].shape[0], : tensors[0].shape[1], : tensors[0].shape[2]] = tensors[0]
    image_sizes = [im.shape[-2:] for im in tensors]

    return ImageList(batched_imgs, image_sizes)

class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)

        image = image.resize(size[::-1], 2)

        if isinstance(target, list):
            target = [t.resize(image.size) for t in target]
        elif target is None:
            return image
        else:
            target = target.resize(image.size)
        return image, target

def TorchToTensor(pic):
    img = torch.from_numpy(np.array(pic, np.uint8, copy=True))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.float32).div(255)

def MegengineToTensor(pic):
    img = megengine.tensor(np.array(pic, np.uint8, copy=True))
    img = img.transpose((2, 0, 1))
    return img.astype(np.float32)/255

def TorchNormalize(image):
    image = image[[2, 1, 0]] * 255
    mean = [102.9801, 115.9465, 122.7717]
    std = [1.0, 1.0, 1.0]
    mean = torch.as_tensor(mean, dtype=torch.float32)
    std = torch.as_tensor(std, dtype=torch.float32)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    image.sub_(mean).div_(std)
    return image

def MegengineNormalize(image):
    image = image[[2, 1, 0]] * 255
    mean = [102.9801, 115.9465, 122.7717]
    std = [1.0, 1.0, 1.0]
    mean = megengine.tensor(mean, dtype=np.float32)
    std = megengine.tensor(std, dtype=np.float32)
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    return (image - mean) / std  


T_Resize = Resize(800, 1333)

# 读取图片
img_root = './data'
img_name_list = os.listdir(img_root)


# pytorch模型初始化
cfg.merge_from_file('official_PAA/configs/paa/paa_R_50_FPN_1.5x.yaml')
model = build_detection_model(cfg)  
model.to(cfg.MODEL.DEVICE)
checkpointer = DetectronCheckpointer(cfg, model)
_ = checkpointer.load('official_PAA/paa_res50.pth')
model.eval()

# megengine模型初始化
megengine_model = get_megengine_hardnet_model(pretrained=True)
megengine_model.eval()

for img_name in img_name_list:
    print(f'inference {img_name}')

    # 1.图片预处理
    img = Image.open(Path(img_root)/img_name).convert("RGB")
    img_resize_torch = T_Resize(img)
    img_resize_megengine = img_resize_torch.copy()

    # 2.图片转Torch.tensor，推理
    torch_start_time = time.time()
    torch_tensor = TorchToTensor(img_resize_torch)
    torch_input = TorchNormalize(torch_tensor)
    torch_input =  to_image_list_torch((torch_input,), 32)
    with torch.no_grad():
        torch_output = model(torch_input.to(cfg.MODEL.DEVICE))
    torch_end_time = time.time()
    print("torch inference time: {:.3f}s".format(torch_end_time-torch_start_time))


    # 3.图片转Megengine.tensor，推理
    megengine_start_time = time.time()
    megengine_tensor = MegengineToTensor(img_resize_megengine)
    megengine_input = MegengineNormalize(megengine_tensor)
    megengine_input = to_image_list_megengine((megengine_input,), 32)
    megengine_output = megengine_model(megengine_input)
    megengine_end_time = time.time()
    print("megengine inference time: {:.3f}s".format(megengine_end_time-megengine_start_time))


    # 4.比较Megengine模型推理结果和官方模型推理结果
    np.testing.assert_allclose(torch_output[0].bbox.detach().cpu().numpy(), megengine_output[0].bbox.numpy(), rtol=1e-3)
    np.testing.assert_allclose(torch_output[0].extra_fields['scores'].detach().cpu().numpy(), megengine_output[0].extra_fields['scores'].numpy(), rtol=1e-3)
    np.testing.assert_allclose(torch_output[0].extra_fields['labels'].detach().cpu().numpy(), megengine_output[0].extra_fields['labels'].numpy(), rtol=1e-3)
    print('pass')

