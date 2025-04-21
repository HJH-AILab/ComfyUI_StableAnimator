
import os
import torch
import numpy as np

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

def load_images_from_folder(folder, ):
        """从目录读取图片"""
        images = []
        files = os.listdir(folder)
        png_files = [f for f in files if f.endswith('.png')]
        # png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        for filename in png_files:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            # img = img.resize((width, height))
            images.append(pil_to_tensor(img))

        return torch.cat(images, dim=0)


import sys
from functools import reduce

FLOAT = ("FLOAT", {"default": 1,
                   "min": -sys.float_info.max,
                   "max": sys.float_info.max,
                   "step": 0.01})

BOOLEAN = ("BOOLEAN", {"default": True})
BOOLEAN_FALSE = ("BOOLEAN", {"default": False})

INT = ("INT", {"default": 1,
               "min": -sys.maxsize,
               "max": sys.maxsize,
               "step": 1})

STRING = ("STRING", {"default": ""})

STRING_ML = ("STRING", {"multiline": True, "default": ""})

STRING_WIDGET = ("STRING", {"forceInput": True})

JSON_WIDGET = ("JSON", {"forceInput": True})

METADATA_RAW = ("METADATA_RAW", {"forceInput": True})

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class FormatData:
    """
    一个用于格式化图像或掩码数据的基础类，支持将不同类型的图像或掩码数据转换为指定的数据类型和形状。
    可以通过FormatImage或FormatMask函数快速切换需要的格式化目标类型和形状。pil作为目标时，shape、dtype参数无效
    """
    valid_types = (torch.Tensor, np.ndarray, Image.Image)
    valid_shapes = ()

    def __init__(self):
        pass

    @classmethod
    def to_np(cls, data, shape=None, dtype=None):
        data = cls._format_data(data, target_type=np.ndarray, shape=shape, dtype=dtype)
        return data

    @classmethod
    def to_torch(cls, data, shape=None, dtype=None):
        data = cls._format_data(data, target_type=torch.Tensor, shape=shape, dtype=dtype)
        return data


    @classmethod
    def _format_data(cls, data, target_type, shape=None, dtype=None):
        if not isinstance(data, cls.valid_types):
            raise TypeError(f"data must be either a numpy.ndarray, a torch.Tensor, a tensorflow.Tensor or a PIL.Image.Image. Got {type(data)} instead.")

        instance = cls()

        if target_type is np.ndarray:
            if isinstance(data, np.ndarray):
                data = instance.convert_dtype(data, dtype)
            elif isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
                data = instance.convert_dtype(data, dtype)
            elif isinstance(data, Image.Image):
                data = ImageOps.exif_transpose(data)
                data = np.array(data)
                data = instance.convert_dtype(data, dtype)
            data = instance._adjust_shape(data, shape)
        elif target_type is torch.Tensor:
            if isinstance(data, torch.Tensor):
                data = data.cpu()
                data = instance._adjust_shape(data, shape)
                data = instance.convert_dtype(data, dtype)
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
                data = instance._adjust_shape(data, shape)
                data = instance.convert_dtype(data, dtype)
            elif isinstance(data, Image.Image):
                data = ImageOps.exif_transpose(data)
                data = data.convert("RGB")
                data = np.array(data)
                data = torch.from_numpy(data)[None,]
                data = instance._adjust_shape(data, shape)
                data = instance.convert_dtype(data, dtype)
        elif target_type is Image.Image:
            images_list = []
            if isinstance(data, Image.Image):
                images_list.append(ImageOps.exif_transpose(data))  # 调整握持方向
            else:
                for img in data if len(data.shape) > 3 or (len(data.shape) == 3 and data.shape[-1] not in [1, 3, 4]) else [data]:
                    if isinstance(img, np.ndarray):
                        img_np = instance.convert_dtype(img, np.uint8)
                    elif isinstance(img, torch.Tensor):
                        img_np = instance.convert_dtype(img.detach().cpu().numpy(), np.uint8)
                    images_list.append(Image.fromarray(img_np))
            # 不需要调用 instance._adjust_shape 方法，也不需要处理 dtype
            if dtype is not None or shape is not None:
                images_list = [img.convert('RGB') for img in images_list]
            data = images_list
        else:
            raise TypeError(f"Unsupported target type: {target_type}")

        return data

    @staticmethod
    def convert_dtype(data, dtype):
        """
        根据目标数据类型转换数据，并确保值在有效范围内。
        """
        # 如果 dtype 为 None，则不转换
        if dtype is None:
            return data

        # 确定当前数据类型
        if isinstance(data, np.ndarray):
            current_dtype = data.dtype
            module = np
        elif isinstance(data, torch.Tensor):
            current_dtype = data.dtype
            module = torch
        else:
            raise TypeError("Unsupported Current Object type conversion dtype.")

        # 根据目标类型进行转换
        if dtype == module.uint8:
            # 根据目标类型进行转换
            if dtype == module.uint8:
                min_val = module.min(data)
                max_val = module.max(data)
            
            if current_dtype in [module.float16, module.float32, module.float64]:
                if 0 <= min_val <= max_val <= 1:
                    data = FormatData._clip_value(data * 255, 0, 255, module.uint8)
                elif -1 <= min_val <= max_val <= 2:
                    data = FormatData._clip_value((data.clip(0, 1) * 255), 0, 255, module.uint8)
                elif 0 <= min_val <= max_val <= 255:
                    data = data.astype(module.uint8) if module is np else data.to(torch.uint8)
                else:
                    data = FormatData._clip_value(data, 0, 255, module.uint8)
            elif current_dtype == module.bool_ if hasattr(module,"bool_") else module.bool:
                data = data.astype(module.uint8) * 255 if module is np else data.to(torch.uint8)
            elif current_dtype == module.uint8:
                pass
            else:
                raise ValueError("Unsupported dtype for conversion to uint8.")
        elif dtype in [module.float16, module.float32, module.float64]:
            min_val = module.min(data)
            max_val = module.max(data)
            
            if current_dtype == module.uint8:
                data = data / 255.0 if module is np else data.float() / 255.0
            elif current_dtype in [module.float16, module.float32, module.float64]:
                if min_val < 0 or max_val > 1:
                    data = FormatData._clip_value(data, 0, 1, dtype)
            elif current_dtype == module.bool:
                data = data.astype(module.float32) * 1.0 if module is np else data.to(torch.float32) * 1.0
            else:
                raise ValueError("Unsupported dtype for conversion to float.")
        elif dtype == module.bool:
            if current_dtype in [module.uint8, module.float16, module.float32, module.float64]:
                data = data > 0 if module is np else data > 0
            elif current_dtype == module.bool:
                pass
            else:
                raise ValueError("Unsupported dtype for conversion to bool.")
        else:
            data = data.astype(dtype) if module is np else data.to(dtype)

        return data

    @staticmethod
    def _clip_value(data, min_val, max_val, dtype):
        """根据数据类型剪切张量/数组的值并将其转换为指定的 dtype."""
        if isinstance(data, np.ndarray):
            return np.clip(data, min_val, max_val).astype(dtype)
        elif isinstance(data, torch.Tensor):
            return torch.clamp(data, min_val, max_val).to(dtype)

    def _transpose(data, axes):
        """
        根据输入张量的类型选择合适的转置方法。
        
        参数:
        - data: 输入的图像或掩码数据。
        - axes: 转置轴。
        """
        if isinstance(data, np.ndarray):
            return data.transpose(axes)
        elif isinstance(data, torch.Tensor):
            return data.permute(axes)
        else:
            raise TypeError(f"Unsupported tensor type: {type(data)}")

    def _adjust_shape(self, data, shape=None):
        """
        调整图像或掩码数据的形状。
        
        参数:
        - data: 输入的图像或掩码数据。
        - shape: 目标形状。
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def check_data(data):
        if isinstance(data, np.ndarray):
            return np.ndarray, data.dtype
        elif isinstance(data, torch.Tensor):
            return torch.Tensor, data.dtype
        elif isinstance(data, Image.Image):
            return Image.Image, data.mode
        else:
            return None, None
        

class FormatImage(FormatData):
    """
    一个用于格式化图像数据的类，支持将不同类型的图像数据转换为指定的数据类型和形状。
    """
    valid_shapes = ['hwc', 'chw', 'bhwc', 'bchw']

    @classmethod
    def _adjust_shape(cls, image, shape='bhwc'):
        """
        调整图像数据的形状。

        参数:
        - image: 输入的图像数据。
        - shape: 目标形状。
        """
        return cls.convert_shape(image, shape)

    @staticmethod
    def check_shape(image):
        """
        检查图像数据的形状，并返回一个描述形状的字符串和高度、宽度的元组。

        参数:
        - image: 输入的图像数据，可以是 numpy 数组、torch 张量、tensorflow 张量或 PIL 图像。

        返回:
        - str: 描述输入数据形状的字符串。
        - tuple: 包含高度和宽度的元组。
        """

        if not isinstance(image, FormatData.valid_types):
            raise TypeError(f"Unsupported image type: {type(image)}")

        if isinstance(image, Image.Image):
            # PIL 图像转换为 numpy 数组，并获取宽度和高度
            width, height = image.size
            return 'wh', (height, width)

        shape = image.shape
        ndim = len(shape)

        # 假设最后一个维度是颜色通道
        channels = shape[-1] if ndim > 2 else 1

        if ndim == 2:
            return 'hw', (shape[0], shape[1])
        elif ndim == 3:
            if channels in [1, 3, 4]:
                return 'hwc', (shape[0], shape[1])
            else:
                return 'bhw', (shape[1], shape[2])
        elif ndim == 4:
            if channels in [1, 3, 4]:
                return 'bhwc', (shape[1], shape[2])
            else:
                return 'bchw', (shape[2], shape[3])
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")

    @staticmethod
    def convert_shape(image, target_shape='bhwc'):
        """
        调整图像数据的形状。

        参数:
        - image: 输入的图像数据。
        - target_shape: 目标形状。

        返回:
        - 调整后的图像数据。
        """
        if target_shape is None:
            target_shape = 'bhwc'
        if target_shape not in FormatImage.valid_shapes:
            raise ValueError(f"Invalid shape: {target_shape}. Supported shapes are {FormatImage.valid_shapes}.")

        src_shape, _ = FormatImage.check_shape(image)
        if src_shape == "bhw":
            src_shape = "chw"  # 假定调用者确定输入的一定是image而不是mask，就把第一个维度看做c而不是b

        # 定义从输入形状到目标形状的映射逻辑
        if src_shape == 'hwc' and target_shape == 'hwc':
            pass  # 已经是 hwc 形状
        elif src_shape == 'hwc' and target_shape == 'chw':
            image = FormatImage._transpose(image, (2, 0, 1))  # 转置到 chw 形状
        elif src_shape == 'hwc' and target_shape == 'bhwc':
            image = image[None, ...]  # 增加批次
        elif src_shape == 'hwc' and target_shape == 'bchw':
            image = FormatImage._transpose(image, (2, 0, 1))[None, ...]  # 转置到 chw 形状然后增加批次

        elif src_shape == 'chw' and target_shape == 'hwc':
            image = FormatImage._transpose(image, (1, 2, 0))  # 转置到 hwc 形状
        elif src_shape == 'chw' and target_shape == 'chw':
            pass  # 已经是 chw 形状
        elif src_shape == 'chw' and target_shape == 'bhwc':
            image = FormatImage._transpose(image, (1, 2, 0))[None, ...]  # 转置到 hwc 形状然后增加批次
        elif src_shape == 'chw' and target_shape == 'bchw':
            image = image[None, ...]  # 增加批次

        elif src_shape == 'bhwc' and target_shape == 'hwc':
            image = image.squeeze(0)  # 去掉批次
        elif src_shape == 'bhwc' and target_shape == 'chw':
            image = FormatImage._transpose(image, (0, 3, 1, 2)).squeeze(0)  # 转置到 chw 形状并去掉批次
        elif src_shape == 'bhwc' and target_shape == 'bhwc':
            pass  # 已经是 bhwc 形状
        elif src_shape == 'bhwc' and target_shape == 'bchw':
            image = FormatImage._transpose(image, (0, 3, 1, 2))  # 转置到 bchw 形状

        elif src_shape == 'bchw' and target_shape == 'hwc':
            image = FormatImage._transpose(image, (0, 2, 3, 1)).squeeze(0)  # 转置到 hwc 形状并去掉批次
        elif src_shape == 'bchw' and target_shape == 'chw':
            image = image.squeeze(0)  # 去掉批次
        elif src_shape == 'bchw' and target_shape == 'bhwc':
            image = FormatImage._transpose(image, (0, 2, 3, 1))  # 转置到 bhwc 形状
        elif src_shape == 'bchw' and target_shape == 'bchw':
            pass  # 已经是 bchw 形状

        elif src_shape == 'hw' and target_shape == 'hwc':
            image = image[..., None]  # 增加末位颜色通道
        elif src_shape == 'hw' and target_shape == 'chw':
            image = image[None, ...]  # 增加首位颜色通道
        elif src_shape == 'hw' and target_shape == 'bhwc':
            image = image[None, ..., None]  # 首尾增加批次和颜色通道
        elif src_shape == 'hw' and target_shape == 'bchw':
            image = image[None, None, ...]  # 前两位增加批次和颜色通道

        else:
            raise ValueError(f"Unsupported conversion from {src_shape} to {target_shape}")

        return image


class FormatMask(FormatData):
    """
    一个用于格式化掩码数据的类，支持将不同类型的掩码数据转换为指定的数据类型和形状。
    """
    valid_shapes = ['hw', 'bhw', 'bhwc', 'bchw']

    @classmethod
    def _adjust_shape(cls, mask, shape='bhw'):
        """
        调整掩码数据的形状。

        参数:
        - mask: 输入的掩码数据。
        - shape: 目标形状。
        """
        return cls.convert_shape(mask, shape)

    @staticmethod
    def check_shape(mask):
        """
        检查掩码数据的形状，并返回一个描述形状的字符串。

        参数:
        - mask: 输入的掩码数据。

        返回:
        - str: 描述输入数据形状的字符串。
        """
        if not isinstance(mask, FormatData.valid_types):
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        if isinstance(mask, Image.Image):
            # PIL 图像转换为 numpy 数组
            mask = np.array(mask)

        shape = mask.shape
        ndim = len(shape)

        # 假设最后一个维度是颜色通道
        channels = mask.shape[-1]

        if ndim == 2:
            return 'hw', shape
        elif ndim == 3:
            if channels in [1, 3, 4]:
                return 'hwc', shape
            else:
                return 'bhw', shape
        elif ndim == 4:
            if channels in [1, 3, 4]:
                return 'bhwc', shape
            else:
                return 'bchw', shape
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")

    @staticmethod
    def convert_shape(mask, target_shape='bhw'):
        """
        调整掩码数据的形状。

        参数:
        - mask: 输入的掩码数据。
        - target_shape: 目标形状。

        返回:
        - 调整后的掩码数据。
        """
        if target_shape is None:
            target_shape = 'bhw'
        if target_shape not in FormatMask.valid_shapes:
            raise ValueError(f"Invalid shape: {target_shape}. Supported Mask shapes are {FormatMask.valid_shapes}.")

        src_shape, _ = FormatMask.check_shape(mask)

        # 定义从输入形状到目标形状的映射逻辑
        if src_shape == 'hw' and target_shape == 'hw':
            pass  # 已经是 hw 形状
        elif src_shape == 'hw' and target_shape == 'bhw':
            mask = mask[None, ...]  # 增加批次
        elif src_shape == 'hw' and target_shape == 'bhwc':
            mask = mask[None, ..., None]  # 首尾增加批次和颜色通道
        elif src_shape == 'hw' and target_shape == 'bchw':
            mask = mask[None, None, ...]  # 增加前两位批次和颜色通道

        elif src_shape == 'bhw' and target_shape == 'hw':
            mask = mask.squeeze(0)  # 去掉首位批次
        elif src_shape == 'bhw' and target_shape == 'bhw':
            pass  # 已经是 bhw 形状
        elif src_shape == 'bhw' and target_shape == 'bhwc':
            mask = mask[..., None]  # 增加颜色通道
        elif src_shape == 'bhw' and target_shape == 'bchw':
            mask = mask[..., None]  # 增加颜色通道
            mask = FormatData._transpose(mask, (0, 3, 1, 2))  # 转置成 bchw 形状

        elif src_shape == 'hwc' and target_shape == 'hw':
            mask = mask.squeeze(-1)  # 去掉末位颜色通道
        elif src_shape == 'hwc' and target_shape == 'bhw':
            mask = mask.squeeze(-1)[None, ...]  # 去掉末位颜色通道并增加批次通道
        elif src_shape == 'hwc' and target_shape == 'bhwc':
            mask = mask[None, ...]  # 增加批次
        elif src_shape == 'hwc' and target_shape == 'bchw':
            mask = FormatData._transpose(mask, (2, 0, 1))[None, ...]  # 转置并增加批次

        elif src_shape == 'bhwc' and target_shape == 'hw':
            mask = mask.squeeze(-1).squeeze(0)  # 去掉末位颜色通道和首位批次
        elif src_shape == 'bhwc' and target_shape == 'bhw':
            mask = mask.squeeze(-1)  # 去掉末位颜色通道
        elif src_shape == 'bhwc' and target_shape == 'bhwc':
            pass  # 已经是 bhwc 形状
        elif src_shape == 'bhwc' and target_shape == 'bchw':
            mask = FormatData._transpose(mask, (0, 3, 1, 2))  # 转置成 bchw 形状

        elif src_shape == 'bchw' and target_shape == 'hw':
            mask = mask.squeeze(0).squeeze(0)  # 去掉前两位批次和颜色通道
        elif src_shape == 'bchw' and target_shape == 'bhw':
            mask = mask.squeeze(1)  # 去掉第二位颜色通道
        elif src_shape == 'bchw' and target_shape == 'bhwc':
            mask = FormatData._transpose(mask, (0, 2, 3, 1))  # 转置成 bhwc 形状
        elif src_shape == 'bchw' and target_shape == 'bchw':
            pass  # 已经是 bchw 形状

        else:
            raise ValueError(f"Unsupported conversion from {src_shape} to {target_shape}")

        return mask


import torch.nn.functional as F
from comfy.utils import lanczos, bislerp  # 导入 comfy.utils 模块

"""当输入图像是PIL的还需要继续debug"""
class Resize:
    valid_methods = ["just_resize", "crop_center", "crop_topleft", "fill_center", "fill_topleft", "align_center", "align_topleft"]
    valid_resamples = ['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos']
    @staticmethod
    def _format_data(data):
        """格式化图像或掩码为bchw形状的torch张量"""
        original_class, original_dtype = FormatData.check_data(data)
        is_mask = False
        original_shape_type = None
        original_shape = None

        if isinstance(data, Image.Image):
            is_mask = False
            original_shape_type = "wh"
            original_shape = data.size
            # data = FormatData._format_data(data, original_class, shape="bchw")
            return data, is_mask, original_shape_type, original_shape, original_dtype

        original_shape = data.shape
        original_dtype = data.dtype
        ndim = len(data.shape)
        
        # 检查是否为mask类型
        if ndim == 2 or (ndim == 3 and data.shape[-1] not in [1, 3, 4]) or data.dtype == torch.bool:
            # 格式化掩码
            is_mask = True
            original_shape_type, _ = FormatMask.check_shape(data)
            data = FormatMask.to_torch(data, shape="bchw")
            # 将布尔类型转为uint8
            if original_dtype == np.bool_:
                data = data.astype(np.uint8)
            if original_dtype == torch.bool:
                data = data.to(torch.uint8)
        else:
            original_shape_type, _ = FormatImage.check_shape(data)
            if not isinstance(original_class, torch.Tensor) or ndim != 4 or data.shape[1] not in [1, 3, 4]:
                # 格式化图像
                data = FormatImage.to_torch(data, shape="bchw")

        return data, is_mask, original_shape_type, original_shape, original_dtype

    @staticmethod
    def _check_resample_method(resample_method):
        """检查插值方法是否有效"""
        valid_methods = ['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos', 'bislerp']  # 添加 bislerp 方法
        if resample_method.lower() not in valid_methods:
            raise ValueError(f"无效的插值方法: {resample_method}. 有效的插值方法包括: {valid_methods}.")

    @staticmethod
    def _resize(data, width, height, resample_method="bicubic"):
        """使用torch和torchvision库调整图像大小"""
        Resize._check_resample_method(resample_method)

        # 选择插值方法
        if resample_method.lower() == "lanczos":
            resized_data = lanczos(data, width, height)
        elif resample_method.lower() == "bislerp":
            resized_data = bislerp(data, width, height)
        else:
            # 对于其他方法，使用 torch 的 interpolate 函数
            resized_data = torch.nn.functional.interpolate(data, size=(height, width), mode=resample_method)

        return resized_data

    @staticmethod
    def _restore_shape_and_dtype(data, is_mask, original_shape_type, original_shape, original_dtype):
        """
        恢复图像或掩码的原始形状和数据类型。

        :param data: 调整大小并填充后的图像或掩码数据。
        :param is_mask: 是否是掩码。
        :param original_shape_type: 原始形状的字符串标识，如 'hw', 'bhw', 'hwc', 'bchw', 'bhwc'。
        :param original_shape: 原始数据的维度数。
        :param original_dtype: 原始数据类型。
        :return: 恢复形状和数据类型的图像或掩码数据。
        """
        # 根据 original_dtype 校正像素值
        if isinstance(original_dtype, torch.dtype):
            data = FormatImage.to_torch(data, original_shape_type, original_dtype) if not is_mask else FormatMask.to_torch(data, original_shape_type, original_dtype)
            if original_dtype == torch.uint8:
                data = torch.clamp(data, 0, 255)
            else:
                data = torch.clamp(data, 0, 1)
        elif isinstance(original_dtype, np.dtype):
            data = FormatImage.to_np(data, original_shape_type, original_dtype) if not is_mask else FormatMask.to_np(data, original_shape_type, original_dtype)
            if original_dtype == np.uint8:
                data = np.clip(data, 0, 255)
            else:
                data = np.clip(data, 0, 1)
        """
        # 根据原始形状恢复数据
        if original_shape_type == 'hw':
            # 单通道掩码，形状为 (h, w)
            data = data.squeeze(0).squeeze(0)
        elif original_shape_type == 'bhw':
            # 批量单通道掩码，形状为 (b, h, w)
            data = data.squeeze(1)
        elif original_shape_type == 'hwc':
            # 单张图像，形状为 (h, w, c)
            data = data.squeeze(0).permute(1, 2, 0)
        elif original_shape_type == 'bchw':
            # 批量图像，形状为 (b, c, h, w)
            data = data
        elif original_shape_type == 'bhwc':
            # 批量图像，形状为 (b, h, w, c)
            data = data.permute(0, 2, 3, 1)
        
        # 恢复数据类型
        if isinstance(original_dtype, torch.dtype):
            data = data.type(original_dtype)
        elif isinstance(original_dtype, np.dtype):
            data = data.astype(original_dtype)
        elif isinstance(original_dtype, tf.DType):
            data = tf.cast(data, original_dtype)
        """
        return data

    @staticmethod
    def just_resize(data, width, height, resample_method="bicubic"):
        """直接调整图像大小"""
        data, is_mask, original_shape_type, original_shape, original_dtype = Resize._format_data(data)
        Resize._check_resample_method(resample_method)
        
        # 调整图像大小
        resized_data = Resize._resize(data, width, height, resample_method)
        
        # 恢复原始形状和数据类型
        resized_data = Resize._restore_shape_and_dtype(resized_data, is_mask, original_shape_type, original_shape, original_dtype)
        
        return resized_data

    @staticmethod
    def crop_topleft(data, width, height, resample_method="bicubic"):
        """
        从图像的左上角开始裁剪，并调整大小到指定的宽度和高度。

        :param data: 输入的图像数据，形状为 (b, c, h, w) 的 Tensor。
        :param width: 目标宽度。
        :param height: 目标高度。
        :param resample_method: 重采样方法，默认为 "bicubic"。
        :return: 裁剪并调整大小后的图像数据，形状为 (b, c, height, width) 的 Tensor。
        """
        data, is_mask, original_shape_type, original_shape, original_dtype = Resize._format_data(data)
        Resize._check_resample_method(resample_method)

        # 获取原始图像的尺寸
        old_width = data.shape[3]
        old_height = data.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height

        # 计算裁剪的宽度和高度
        if old_aspect > new_aspect:
            # 宽度大于目标宽高比，需要裁剪宽度
            crop_width = int(old_height * new_aspect)
            cropped_data = data[:, :, :, :crop_width]
        elif old_aspect < new_aspect:
            # 高度大于目标宽高比，需要裁剪高度
            crop_height = int(old_width / new_aspect)
            cropped_data = data[:, :, :crop_height, :]
        else:
            # 宽高比相同，不需要裁剪
            cropped_data = data

        # 调整图像大小
        resized_data = Resize._resize(cropped_data, width, height, resample_method)

        # 恢复原始形状和数据类型
        resized_data = Resize._restore_shape_and_dtype(resized_data, is_mask, original_shape_type, original_shape, original_dtype)

        return resized_data

    @staticmethod
    def crop_bottomleft(data, width, height, resample_method="bicubic"):
        """
        从图像的左下角开始裁剪，并调整大小到指定的宽度和高度。

        :param data: 输入的图像数据，形状为 (b, c, h, w) 的 Tensor。
        :param width: 目标宽度。
        :param height: 目标高度。
        :param resample_method: 重采样方法，默认为 "bicubic"。
        :return: 裁剪并调整大小后的图像数据，形状为 (b, c, height, width) 的 Tensor。
        """
        data, is_mask, original_shape_type, original_shape, original_dtype = Resize._format_data(data)
        Resize._check_resample_method(resample_method)

        # 获取原始图像的尺寸
        old_width = data.shape[3]
        old_height = data.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height

        # 计算裁剪的宽度和高度
        if old_aspect > new_aspect:
            # 宽度大于目标宽高比，需要裁剪宽度
            crop_width = int(old_height * new_aspect)
            # 从左边界往右裁剪 crop_width 个像素
            cropped_data = data[:, :, :, :crop_width]
        elif old_aspect < new_aspect:
            # 高度大于目标宽高比，需要裁剪高度
            crop_height = int(old_width / new_aspect)
            # 从底部边界往上裁剪 crop_height 个像素
            cropped_data = data[:, :, crop_height:, :]
        else:
            # 宽高比相同，不需要裁剪
            cropped_data = data

        # 调整图像大小
        resized_data = Resize._resize(cropped_data, width, height, resample_method)

        # 恢复原始形状和数据类型
        resized_data = Resize._restore_shape_and_dtype(resized_data, is_mask, original_shape_type, original_shape, original_dtype)

        return resized_data

    @staticmethod
    def crop_center(data, width, height, resample_method="bicubic"):
        """先根据目标宽高比裁剪中心部分，再调整大小"""
        data, is_mask, original_shape_type, original_shape, original_dtype = Resize._format_data(data)
        Resize._check_resample_method(resample_method)
        
        # 获取原始图像的尺寸
        old_width = data.shape[3]
        old_height = data.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height

        # 计算裁剪的起始位置
        x = 0
        y = 0
        if old_aspect > new_aspect:
            # 宽度大于目标宽高比，需要裁剪两侧
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            # 高度大于目标宽高比，需要裁剪上下
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)

        # 裁剪中心部分
        cropped_data = data[:, :, y:old_height - y, x:old_width - x]

        # 调整图像大小
        resized_data = Resize._resize(cropped_data, width, height, resample_method)

        # 恢复原始形状和数据类型
        resized_data = Resize._restore_shape_and_dtype(resized_data, is_mask, original_shape_type, original_shape, original_dtype)
        
        return resized_data

    @staticmethod
    def fill_topleft(data, width, height, resample_method="bicubic"):
        """
        调整大小并将图像填充到与目标左上角对齐的位置。
        """
        data, is_mask, original_shape_type, original_shape, original_dtype = Resize._format_data(data)
        Resize._check_resample_method(resample_method)

        # 计算缩放因子
        scale_factor_w = width / data.shape[3]
        scale_factor_h = height / data.shape[2]
        scale_factor = min(scale_factor_w, scale_factor_h)

        # 计算调整后的尺寸
        resized_width = int(data.shape[3] * scale_factor)
        resized_height = int(data.shape[2] * scale_factor)

        # 调整图像大小
        resized_data = Resize._resize(data, resized_width, resized_height, resample_method)

        # 创建填充后的图像
        filled_data = resized_data.clone()

        # 计算需要填充的尺寸
        pad_left = 0
        pad_top = 0
        pad_right = max(width - resized_data.shape[3], 0)
        pad_bottom = max(height - resized_data.shape[2], 0)

        # 使用 F.pad 进行填充
        padding_params = [pad_left, pad_right, pad_top, pad_bottom]  # [left, right, top, bottom]
        filled_data = F.pad(filled_data, padding_params, mode='constant', value=0)

        # 恢复原始形状和数据类型
        filled_data = Resize._restore_shape_and_dtype(filled_data, is_mask, original_shape_type, original_shape, original_dtype)

        return filled_data

    @staticmethod
    def fill_center(data, width, height, resample_method="bicubic"):
        """调整大小并填充至目标中心"""
        data, is_mask, original_shape_type, original_shape, original_dtype = Resize._format_data(data)
        Resize._check_resample_method(resample_method)

        # 计算缩放因子
        scale_factor_w = width / data.shape[3]
        scale_factor_h = height / data.shape[2]
        scale_factor = min(scale_factor_w, scale_factor_h)

        # 计算调整后的尺寸
        resized_width = int(data.shape[3] * scale_factor)
        resized_height = int(data.shape[2] * scale_factor)

        # 调整图像大小
        resized_data = Resize._resize(data, resized_width, resized_height, resample_method)

        # 创建填充后的图像
        filled_data = resized_data.clone()

        # 计算需要填充的尺寸
        pad_left = (width - resized_data.shape[3]) // 2
        pad_top = (height - resized_data.shape[2]) // 2

        # 如果宽度不够，则在左右两侧添加零填充
        if width > resized_data.shape[3]:
            left_pad = torch.zeros((data.shape[0], data.shape[1], resized_data.shape[2], pad_left), dtype=data.dtype)
            right_pad = torch.zeros((data.shape[0], data.shape[1], resized_data.shape[2], width - resized_data.shape[3] - pad_left), dtype=data.dtype)
            filled_data = torch.cat((left_pad, filled_data, right_pad), dim=3)

        # 如果高度不够，则在上下两侧添加零填充
        if height > resized_data.shape[2]:
            top_pad = torch.zeros((data.shape[0], data.shape[1], pad_top, filled_data.shape[3]), dtype=data.dtype)
            bottom_pad = torch.zeros((data.shape[0], data.shape[1], height - resized_data.shape[2] - pad_top, filled_data.shape[3]), dtype=data.dtype)
            filled_data = torch.cat((top_pad, filled_data, bottom_pad), dim=2)

        # 恢复原始形状和数据类型
        filled_data = Resize._restore_shape_and_dtype(filled_data, is_mask, original_shape_type, original_shape, original_dtype)

        return filled_data

    @staticmethod
    def align_topleft(data, width, height, resample_method=None):
        """
        在原图右侧和下方进行填充以达到目标宽高，如果原图宽或高大于目标宽高，则执行裁剪。

        :param data: 输入的图像数据，形状为 (b, c, h, w) 的 Tensor。
        :param resample_method: 可选参数，目前未使用。
        """
        data, is_mask, original_shape_type, original_shape, original_dtype = Resize._format_data(data)

        current_height, current_width = data.shape[2], data.shape[3]

        # 如果原图宽或高大于目标宽高，则执行裁剪
        if current_width > width:
            data = data[:, :, :, :max(width, 1)]
        if current_height > height:
            data = data[:, :, :max(height, 1), :]

        current_height, current_width = data.shape[2], data.shape[3]

        pad_right = max(width - current_width, 0)
        pad_bottom = max(height - current_height, 0)

        # 计算填充参数
        padding_params = [0, pad_right, 0, pad_bottom]  # [left, right, top, bottom]

        # 使用 F.pad 进行填充
        data = F.pad(data, padding_params, mode='constant', value=0)

        aligned_data = Resize._restore_shape_and_dtype(data, is_mask, original_shape_type, original_shape, original_dtype)

        return aligned_data

    @staticmethod
    def align_center(data, width, height, resample_method=None):
        """
        在原图像四周进行填充以达到目标宽高，如果原图宽或高大于目标宽高，则执行裁剪。

        :param data: 输入的图像数据，形状为 (b, c, h, w) 的 Tensor。
        :param width: 目标宽度。
        :param height: 目标高度。
        :param resample_method: 可选参数，目前未使用。
        :return: 填充后的图像数据，形状为 (b, c, height, width) 的 Tensor。
        """
        data, is_mask, original_shape_type, original_shape, original_dtype = Resize._format_data(data)

        current_height, current_width = data.shape[2], data.shape[3]

        # 如果原图宽或高大于目标宽高，则执行裁剪
        if current_width > width:
            # 计算从两侧均匀裁剪的起始位置
            start_col = (current_width - width) // 2
            end_col = start_col + width
            data = data[:, :, :, start_col:end_col]
        if current_height > height:
            # 计算从两侧均匀裁剪的起始位置
            start_row = (current_height - height) // 2
            end_row = start_row + height
            data = data[:, :, start_row:end_row, :]

        current_height, current_width = data.shape[2], data.shape[3]

        # 计算需要填充的尺寸
        pad_left = max((width - current_width) // 2, 0)
        pad_right = max(width - current_width - pad_left, 0)
        pad_top = max((height - current_height) // 2, 0)
        pad_bottom = max(height - current_height - pad_top, 0)

        # 计算填充参数
        padding_params = [pad_left, pad_right, pad_top, pad_bottom]

        # 使用 F.pad 进行填充
        data = F.pad(data, padding_params, mode='constant', value=0)

        aligned_data = Resize._restore_shape_and_dtype(data, is_mask, original_shape_type, original_shape, original_dtype)

        return aligned_data


# 创建模型简化映射字典和访问器
class SimpMapAccessor:
    def __init__(self, data_structure):
        self.name = "SimpMapAccessor"
        is_dict = self._extract_is_dict(data_structure.copy())
        self._mapping = self._prepare_mapping(data_structure.copy(), is_dict)
        self._accessor = self._create_accessor(is_dict)

    def _extract_is_dict(self, data_structure):
        is_dict = False
        if 'is_dict' in data_structure:
            is_dict = data_structure.pop('is_dict')
        else:
            print(f"{self.name} No 'is_dict' found in data structure, using default False.")
        return is_dict

    def _prepare_mapping(self, data_structure, is_dict):
        orig_get = []
        simp_get = []
        if 'custom_map' in data_structure:
            custom_map = data_structure.pop('custom_map')
            for simp_key, orig_key in custom_map.items():
                simp_get.append(simp_key)
                orig_get.append(orig_key)
        self._build_map(data_structure, orig_get, "", simp_get, "", is_dict)
        return dict(zip(simp_get, orig_get))

    def _build_map(self, data, orig_get, prefix_orig, simp_get, prefix_simp, is_dict=False):
        for key, value in data.items():
            if isinstance(value, dict):
                new_simp = f"{prefix_simp}{self._simplify_obj_name(simp_get, key)}"
                new_orig = f"{prefix_orig},{key}" if prefix_orig else f"{key}"
                self._build_map(value, orig_get, new_orig, simp_get, new_simp, is_dict)
            else:
                simple_key = self._simplify_attr_name(key)
                simp_get.append(f"{prefix_simp}{simple_key}")
                orig_get.append(f"{prefix_orig},{key}" if prefix_orig else f"{key}")

    def _simplify_obj_name(self, simp_key, key):
        simplified = ""
        index = 0
        while True:
            if index < len(key):
                char = key[index]
                temp_simplified = simplified + char
                if temp_simplified not in simp_key:
                    simplified = temp_simplified
                    break
                index += 1
            else:
                raise ValueError(f"Could not find unique simplified name for key {key}.")
        return f'{simplified}_'

    def _simplify_attr_name(self, key):
        if '_' in key and key.rfind('_') != len(key) - 1:
            return key.rsplit('_', 1)[-1]
        else:
            return key

    def _create_accessor(self, is_dict):
        if is_dict:
            return self._create_dict_accessor
        else:
            return self._create_object_accessor

    def _create_object_accessor(self, parts):
        def object_accessor(r):
            try:
                return reduce(getattr, parts.split(','), r)
            except AttributeError:
                print(f"Failed to access attribute path: {'.'.join(parts.split(','))}")
                return None
        return object_accessor

    def _create_dict_accessor(self, parts):
        def dict_accessor(r):
            try:
                for part in parts.split(','):
                    part = part.strip("['").strip("']")
                    if isinstance(r, dict) and part in r:
                        r = r[part]
                    else:
                        print(f"Key '{part}' not found in dictionary.")
                        return None
                return r
            except KeyError:
                print(f"Failed to access key path: {'.'.join(parts.split(','))}")
                return None
        return dict_accessor


# 张量类型转换
def to_torch(tensor, dtype):
    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).to(dtype)
    else:
        return tensor.to(dtype)

def to_np(tensor, dtype):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().astype(dtype)
    else:
        return tensor.astype(dtype)

def align_properties(target_tensor, reference_tensor):
    """
    验证并转换输入对象的张量类型、设备和数据类型。

    参数:
    target_tensor: 第一个张量对象，用于确定目标类型、设备和数据类型。
    reference_tensor: 第二个张量对象，将被转换为与目标一致。

    返回:
    Tuple[Any, Any]: 验证并转换后的两个对象。
    
    其他说明:
    TensorFlow 的张量在设计上不直接支持设备的概念，而是通过设备上下文（如 tf.device）来管理设备。因此，在 to_tf 函数中不需要也不应该直接处理设备参数。TensorFlow 会自动管理张量的设备分配，通常不需要手动指定设备。
    """
    # 判断 target_tensor 的类型并进行相应处理
    if isinstance(target_tensor, torch.Tensor):
        target_device = target_tensor.device
        target_dtype = target_tensor.dtype
        convert_func = to_torch
    elif isinstance(target_tensor, np.ndarray):
        target_device = None
        target_dtype = target_tensor.dtype
        convert_func = to_np
    else:
        raise TypeError("target_tensor must be either a numpy.ndarray, a torch.Tensor or a tensorflow.Tensor.")

    # 判断 reference_tensor 的类型并进行相应处理
    if not isinstance(reference_tensor, (torch.Tensor, np.ndarray,)):
        raise TypeError("reference_tensor must be either a numpy.ndarray, a torch.Tensor or a tensorflow.Tensor.")

    # 根据 target_tensor 的类型调整 reference_tensor
    reference_tensor = convert_func(reference_tensor, target_dtype)
    # 对于 PyTorch 张量，还需要对设备进行对齐
    if isinstance(target_tensor, torch.Tensor) and isinstance(reference_tensor, torch.Tensor):
        reference_tensor = reference_tensor.to(device=target_device)

    return target_tensor, reference_tensor

def get_shape(tensor):
    """
    获取张量的形状。
    
    参数:
        tensor: 输入张量，可以是 numpy.ndarray、torch.Tensor、tf.Tensor 或 PIL.Image.Image。
    
    返回:
        tuple: 张量的形状。
    """
    if isinstance(tensor, Image.Image):
        return tensor.size[::-1]
    elif isinstance(tensor, np.ndarray):
        return tensor.shape
    elif isinstance(tensor, torch.Tensor):
        return tensor.shape

def pad_tensor_shape(tensor, target_shape):
    """
    对张量进行填充以匹配目标形状。
    
    参数:
        tensor: 输入张量，可以是 numpy.ndarray、torch.Tensor、tf.Tensor 或 PIL.Image.Image。
        target_shape (tuple): 目标形状。
    
    返回:
        同类型张量: 对齐后的张量。
    """
    if isinstance(tensor, Image.Image):
        # PIL 图像的填充
        current_shape = get_shape(tensor)
        if current_shape == target_shape:
            return tensor
        padded_image = Image.new('RGB', (target_shape[0], target_shape[1]))
        padded_image.paste(tensor, (0, 0))
        return padded_image
    elif isinstance(tensor, np.ndarray):
        # Numpy 数组的填充
        current_shape = get_shape(tensor)
        if current_shape == target_shape:
            return tensor
        pad_width = [(0, target_shape[i] - current_shape[i]) for i in range(len(current_shape))]
        return np.pad(tensor, pad_width, mode='constant')
    elif isinstance(tensor, torch.Tensor):
        # PyTorch 张量的填充
        current_shape = get_shape(tensor)
        if current_shape == target_shape:
            return tensor
        pad_width = [0 for _ in range(len(current_shape) * 2)]
        for i in range(len(current_shape)):
            if current_shape[i] < target_shape[i]:
                pad_width[i * 2 + 1] = target_shape[i] - current_shape[i]  # padding right/bottom
        return F.pad(tensor, pad_width)

def align_shapes(tensor1, tensor2):
    """
    对齐两个张量的形状，使得它们在所有维度上具有相同的大小。
    如果一个张量在某个维度上的大小小于另一个张量，则该函数会在必要的维度上扩展该张量以匹配另一个张量的大小。
    
    参数:
        tensor1: 第一个张量，可以是 numpy.ndarray、torch.Tensor、tf.Tensor 或 PIL.Image.Image。
        tensor2: 第二个张量，可以是 numpy.ndarray、torch.Tensor、tf.Tensor 或 PIL.Image.Image。
    
    返回:
        tuple: 包含对齐后的两个张量 (aligned_tensor1, aligned_tensor2)。
    """

    # 确保输入张量的类型为 numpy.ndarray、torch.Tensor、tf.Tensor 或 PIL.Image.Image
    if not (isinstance(tensor1, (np.ndarray, torch.Tensor, Image.Image))):
        raise ValueError("Unsupported type for tensor1.")
    if not (isinstance(tensor2, (np.ndarray, torch.Tensor, Image.Image))):
        raise ValueError("Unsupported type for tensor2.")

    # 获取两个张量的形状
    shape1 = get_shape(tensor1)
    shape2 = get_shape(tensor2)

    # 检查维度数量是否相同
    if len(shape1) != len(shape2):
        raise ValueError("The number of dimensions must be the same for both tensors.")

    # 计算两个张量的最大形状
    max_shape = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))

    # 对齐张量1
    aligned_tensor1 = pad_tensor_shape(tensor1, max_shape)

    # 对齐张量2
    aligned_tensor2 = pad_tensor_shape(tensor2, max_shape)

    return aligned_tensor1, aligned_tensor2

def align_batch(tensor1, tensor2, mode='pad'):
    """
    对齐两个带有 batch 维度的张量，使得它们的 batch 数量相同。
    如果一个张量的 batch 数量小于另一个张量，则该函数会在必要的维度上扩展该张量以匹配另一个张量的 batch 数量（pad模式），
    或者截断较长的张量以匹配较短的张量的 batch 数量（truncate模式）。

    参数:
        tensor1 (torch.Tensor): 第一个张量，形状为 (batch_size1, ...)
        tensor2 (torch.Tensor): 第二个张量，形状为 (batch_size2, ...)
        mode (str): 对齐模式，支持 'pad' 和 'truncate'，默认为 'pad'。

    返回:
        tuple: 包含对齐后的两个张量 (aligned_tensor1, aligned_tensor2)。
    """

    # 获取两个张量的 batch 维度大小
    batch_size1 = tensor1.shape[0]
    batch_size2 = tensor2.shape[0]

    # 计算最大/最小 batch 维度大小
    max_batch_size = max(batch_size1, batch_size2)
    min_batch_size = min(batch_size1, batch_size2)

    # 根据模式选择对齐方式
    if mode == 'pad':
        # 对齐张量1
        if batch_size1 < max_batch_size:
            aligned_tensor1 = pad_tensor_shape(tensor1, (max_batch_size,) + tensor1.shape[1:])
        else:
            aligned_tensor1 = tensor1

        # 对齐张量2
        if batch_size2 < max_batch_size:
            aligned_tensor2 = pad_tensor_shape(tensor2, (max_batch_size,) + tensor2.shape[1:])
        else:
            aligned_tensor2 = tensor2

    elif mode == 'truncate':
        # 截断张量1
        if batch_size1 > min_batch_size:
            aligned_tensor1 = tensor1[:min_batch_size]
        else:
            aligned_tensor1 = tensor1

        # 截断张量2
        if batch_size2 > min_batch_size:
            aligned_tensor2 = tensor2[:min_batch_size]
        else:
            aligned_tensor2 = tensor2

    else:
        raise ValueError(f"Unsupported alignment mode '{mode}'. Supported modes are 'pad' and 'truncate'.")

    return aligned_tensor1, aligned_tensor2

def batch_dimension_to_list(batch_item, keep_shape=True):
    if keep_shape:
        item_list = [torch.unsqueeze(i, dim=0) for i in batch_item.clone()]  # list中的元素从(b,)转为(1,)，形状不变
    else:
        item_list = [i.squeeze() for i in batch_item.clone()]  # list中的元素从(b,)转为(1,)，形状改变，失去batch维度
    
    return item_list

def list_to_batch_dimension(item_list, mode="concat"):
    if not isinstance(item_list, list):
        return item_list

    all_tensors = all(isinstance(item, torch.Tensor) for item in item_list)
    new_list = []

    if all_tensors:
        if mode == "concat":
            # 对于 concat 模式，直接连接张量，不需要填充
            return torch.cat(item_list, dim=0)
        elif mode == "stack":
            # 对于 stack 模式，需要确保所有张量的形状完全相同
            max_shape = tuple(max(dim) for dim in zip(*[tensor.shape for tensor in item_list]))
            for tensor in item_list:
                padding = []
                for dim, max_dim in zip(tensor.shape, max_shape):
                    padding.append((0, max_dim - dim))
                padding = sum(padding[::-1], ())
                new_tensor = torch.nn.functional.pad(tensor, padding)
                new_list.append(new_tensor)
            return torch.stack(new_list, dim=0)
        else:
            raise ValueError(f"Invalid mode. Must be either 'concat' or 'stack'.")
    else:
        element_types = set(type(item) for item in item_list)
        if len(element_types) > 1:
            raise ValueError("When not all elements are tensors, the non-tensor elements should be of the same type.")
        elif element_types == {int} or element_types == {float}:
            return torch.tensor(item_list)
        elif element_types == {np.ndarray}:
            max_length = 0
            for arr in item_list:
                max_length = max(max_length, len(arr))
            for arr in item_list:
                padding_length = max_length - len(arr)
                if padding_length > 0:
                    padding_array = np.zeros(padding_length, dtype=arr.dtype)
                    new_arr = np.concatenate((arr, padding_array))
                    new_list.append(new_arr)
                else:
                    new_list.append(arr)
            return np.stack(new_list)
        else:
            raise ValueError("Unsupported non-tensor element types.")


'''
以下是格式化颜色（Color）的方法
'''
def normalize_color(color):
    """
    将颜色转换为标准的整数格式。
    参数:
    - color: 颜色，可以是 (R, G, B) 或 (R, G, B, A) 的浮点数形式，
             十六进制字符串（例如 '#RRGGBB' 或 '#AARRGGBB'），或者 (R, G, B) 的整数形式。
    返回:
    - color: 转换后的颜色，格式为 (R, G, B) 或 (R, G, B, A) 的整数形式。
    """
    if isinstance(color, tuple):  # 处理 (R, G, B) 或 (R, G, B, A) 形式的颜色
        if all(isinstance(c, int) and 0 <= c <= 255 for c in color):  # 整数格式
            return color
        elif all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in color):  # 浮点数格式
            color = tuple(int(round(float(c) * 255)) for c in color)  # 确保所有值都是浮点数
            return color
        else:
            raise ValueError("Invalid color value range.")

    elif isinstance(color, str):  # 处理十六进制颜色字符串
        if color.startswith('#'):
            color = color[1:]  # 移除开头的 '#'

        if len(color) == 6:  # RGB
            color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        elif len(color) == 8:  # RGBA
            color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4, 6))
        else:
            raise ValueError("Invalid hexadecimal color format.")

    else:
        raise TypeError("Unsupported color type.")

    return color

def format_float_color(color):
    color = normalize_color(color)
    return tuple(c / 255.0 for c in color)

def format_int_color(color):
    return normalize_color(color)


'''
以下是一些数学（Math）算法
'''
import math

def gcd(a, b):
    """计算最大公约数"""
    while b:
        a, b = b, a % b
    return a

def simplest_ratio(a, b):
    """计算a、b的的最简比。即通过计算a、b的最大公约数，使其互质"""
    divisor = gcd(a, b)
    return a // divisor, b // divisor

def closest_multiple(value, multiple=32):
    """计算最接近的整倍数。比较上下两个整数值的方法避免了浮点运算精度和性能可能出现的问题。"""
    if multiple == 0:
        raise ValueError("Multiple cannot be zero.")
    
    if not isinstance(value, (int, float)) or not isinstance(multiple, int):
        raise TypeError("Both value and multiple must be numbers.")

    lower = (value // multiple) * multiple # 向下取整
    upper = ((value + multiple - 1) // multiple) * multiple # 向上取整
    # 找到最接近的值
    if upper - value < value - lower:
        return upper
    else:
        return lower

def best_step(target, base=640, step_unit=64, max_steps=6):
    """
    根据给定的目标值、基数、步长单位和最大步数，计算最接近目标值的步数及其对应的步长值。

    参数:
        target (int): 目标值。
        base (int): 基数值，默认为640。
        step_unit (int): 步长单位，默认为64。
        max_steps (int): 最大步数，默认为6。

    返回:
        tuple: 包含三个元素的元组：
            best_steps (int): 最佳步数。
            best_step_value (int): 对应的最佳步长值。
            min_offset (int): 实际值与目标值之间的最小偏移量。
    """
    # 计算目标值和基数之间的差值
    diff = target - base
    
    # 计算差值除以步长单位的商（可以是小数）
    total_steps = diff / step_unit
    
    # 在最大步数到 2（最小步数）之间，找到使实际值与目标值之间距离最小的步数和步长值
    best_result = None
    min_distance = float('inf')
    best_steps = None
    for steps in range(2, max_steps + 1):
        # 计算候选步长值
        candidate_value = round(total_steps / steps) * step_unit
        
        # 计算候选步长值与目标值之间的距离
        distance = target - (base + candidate_value * steps)
        
        # 如果当前距离更小，则更新最佳结果
        if abs(distance) < abs(min_distance):
            min_distance = distance
            best_result = (steps, candidate_value)
            best_steps = steps
        elif abs(distance) == abs(min_distance):
            # 如果距离相同，则取较大的步数
            if steps > best_steps:
                best_result = (steps, candidate_value)
                best_steps = steps
    
    # 解析最佳结果
    best_steps, best_step_value = best_result
    
    # 计算实际值与目标值之间的偏移量
    min_offset = base + best_step_value * best_steps - target
    
    return best_steps, best_step_value, min_offset

def short_side_to(width, height, target_short_side=640, multiple=32):
    """
    根据给定的宽度和高度，按比例缩放以使短边等于目标短边长度，并确保最终的长边长度是特定整数倍数。

    参数:
        width (int): 图像的宽度。
        height (int): 图像的高度。
        target_short_side (int): 目标短边长度，默认为 640。
        multiple (int): 长边长度需要是此数的整数倍，默认为 32。

    返回:
        tuple: 包含目标宽度和高度的元组 (target_width, target_height)。
    """
    # 确定短边和长边
    short_side = min(width, height)
    long_side = max(width, height)

    # 计算缩放因子
    scale_factor = target_short_side / short_side

    # 缩放长边
    scaled_long_side = int(long_side * scale_factor)

    # 确保长边是 multiple 的倍数
    target_long_side = int(np.floor(scaled_long_side / multiple) * multiple)

    # 根据长边确定目标宽度和高度
    if width == short_side:
        target_width = target_short_side
        target_height = target_long_side
    else:
        target_width = target_long_side
        target_height = target_short_side

    return target_width, target_height

def calc_target_size_by_aspect(source_width, source_height, target_resolution, aspect_ratio=None, factor=1):
    """
    根据给定的分辨率（像素数量）和宽高比计算目标尺寸。

    :param source_width: 原始图像宽度
    :param source_height: 原始图像高度
    :param target_resolution: 目标分辨率 (像素数量)
    :param aspect_ratio: 字符串形式的宽高比，例如 "16:9"；如果为 None，则使用原始宽高比
    :param factor: 目标尺寸的宽度和高度应该是该值的整数倍，默认为 1，最大不超过 1024
    :return: 目标宽度和高度
    """
    # 校验 aspect_ratio 是否合法
    if aspect_ratio is not None:
        if not isinstance(aspect_ratio, str) or ':' not in aspect_ratio:
            raise ValueError("Aspect ratio must be a string containing a colon (:) and two integers.")
        ratio_parts = aspect_ratio.split(':')
        if len(ratio_parts) != 2 or not all(part.replace('.', '', 1).isdigit() for part in ratio_parts):
            raise ValueError("Aspect ratio must be in the format 'long_side:short_side'.")

        # 将字符串转换为浮点数并四舍五入为整数
        long_side = int(round(float(ratio_parts[0])))
        short_side = int(round(float(ratio_parts[1])))

        # 确保 long_side 是较大的那个数，short_side 是较小的那个数
        if long_side < short_side:
            long_side, short_side = short_side, long_side

        # 计算 long_side 和 short_side 的最简比
        # long_side, short_side = simplest_ratio(long_side, short_side)

        # 限制 long_side / short_side 的比例
        if long_side / short_side > 1000:
            raise ValueError("The aspect ratio cannot exceed 1000:1.")
    else:
        # 使用原始图像的宽高比
        long_side, short_side = max(source_width, source_height), min(source_width, source_height)
        aspect_ratio = f"{long_side}:{short_side}"

        # 确保 long_side 和 short_side 互质
        long_side, short_side = simplest_ratio(long_side, short_side)

    # 计算 unit_length
    adjusted_resolution = int(round(target_resolution / (factor ** 2)))
    unit_length = (adjusted_resolution / (long_side * short_side)) ** 0.5

    # 如果指定了倍数，则确保单位长度是该倍数的整数倍
    if factor < 1 or factor > 1024 or not isinstance(factor, int):
        raise ValueError("Factor must be a positive integer between 1 and 1024.")

    # 计算目标长边和短边
    target_long_side = int(round(long_side * unit_length)) * factor
    target_short_side = int(round(short_side * unit_length)) * factor

    # 确保目标宽度和高度至少为 factor
    target_long_side = max(target_long_side, factor)
    target_short_side = max(target_short_side, factor)

    # 根据原始图像的方向确定目标宽度和高度
    if source_width >= source_height:
        target_width = target_long_side
        target_height = target_short_side
    else:
        target_width = target_short_side
        target_height = target_long_side

    return target_width, target_height


    """
    验证输入数据的类型和形状，并返回原始数据类型和拆包后的数据。

    参数:
    data: 输入数据，形状为 (n, 2) 或 (2,) 的张量、numpy 数组、列表或元组。
    name: 输入数据的名称，用于错误信息。

    返回:
    data_x: 拆包后的 X 坐标。
    data_y: 拆包后的 Y 坐标。
    original_type: 原始数据类型。
    original_ndim: 原始数据的维度。
    """
    original_type = None
    if isinstance(data, (list, tuple)):
        # 转换为 numpy 数组
        data = np.array(data)
        original_type = type(data)
    elif isinstance(data, np.ndarray):
        data = data
        original_type = np.ndarray
    elif isinstance(data, torch.Tensor):
        data = data
        original_type = torch.Tensor
    else:
        raise ValueError(f"{name} 必须是 torch 张量、numpy 数组、列表或元组")

    # 检查数据的维度和形状
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{name} 必须是形状为 (n, 2) 或 (2,) 的张量、numpy 数组、列表或元组")
    elif data.ndim == 2 and data.shape[1] == 2:
        original_ndim = 2
    elif data.ndim == 1 and data.shape[0] == 2:
        original_ndim = 1

    # 返回拆包后的数据和原始数据类型
    return data[..., 0], data[..., 1], original_type, original_ndim