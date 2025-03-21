# Midas Depth Estimation
# From https://github.com/isl-org/MiDaS
# MIT LICENSE

import cv2
import numpy as np
import torch
import PIL.Image as Image
from einops import rearrange
from .api import MiDaSInference
from image_gen_aux import DepthPreprocessor

class MidasDetector:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()     

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]
            print(depth,depth.shape)

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            return depth_image, normal_image

class DPTDetector:
    def __init__(self):
        self.processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3
        summ=0
        is_depth=False
        for i in range(10):
            for j in range(10):
                summ+=abs(int(input_image[i,j,0])-int(input_image[i,j,1]))
        if summ<10:
            is_depth=True
            
        image_depth = Image.fromarray(input_image.astype(np.uint8))
        with torch.no_grad():
            if is_depth:
                print("Condition Image is depth!")
                depth = np.array(image_depth).astype(np.float32)
            else:
                depth = np.array(self.processor(image_depth)[0]).astype(np.float32)
            depth = torch.from_numpy(depth).float().cuda()
            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            return depth_image, normal_image
