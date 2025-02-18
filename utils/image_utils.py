
import os
import torch
import numpy as np
import cv2

from PIL import ImageOps, Image

def pil_to_tensor(image_pil):
    i = ImageOps.exif_transpose(image_pil)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image)[None,]
    return image_tensor

def tensor_to_pil(image_tensor):
    image = image_tensor[0]
    i = 255. * image.cpu().numpy()
    image_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
    return image_pil

def tensor_to_np(image_tensor):
    image = image_tensor[0]
    i = 255. * image.cpu().numpy()
    image_np = np.clip(i, 0, 255).astype(np.uint8)
    return image_np

def np_to_tensor(image_np):
    image = image_np.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image)[None,]
    return image_tensor

def load_images_from_folder(folder, width, height):
        """从目录读取图片"""
        images = []
        files = os.listdir(folder)
        png_files = [f for f in files if f.endswith('.png')]
        png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        for filename in png_files:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img = img.resize((width, height))
            images.append(img)

        return images

def save_frames_as_mp4(frames, output_mp4_path, fps):
    print("Starting saving the frames as mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'H264' for better quality
    out = cv2.VideoWriter(output_mp4_path, fourcc, float(fps), (width, height))
    for frame in frames:
       frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       out.write(frame_bgr)
    out.release()