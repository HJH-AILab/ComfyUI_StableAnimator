import os
import torch
import numpy as np
import time

from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

import folder_paths

from .StableAnimator.animation.modules.attention_processor import AnimationAttnProcessor
from .StableAnimator.animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
# from .StableAnimator.animation.modules.face_model import FaceModel
from .utils.face_model import FaceModel #这里重写StableAnimator.animation.modules.face_model.FaceModel 类，以适应ComfyUI的模型加载机制
from .StableAnimator.animation.modules.id_encoder import FusionFaceId
from .StableAnimator.animation.modules.pose_net import PoseNet
from .StableAnimator.animation.modules.unet import UNetSpatioTemporalConditionModel
from .StableAnimator.animation.pipelines.inference_pipeline_animation import InferenceAnimationPipeline

from .utils.image_utils import tensor_to_pil, tensor_to_np, np_to_tensor, load_images_from_folder, save_frames_as_mp4

GLOBAL_CATEGORY = "HJH_StableAnimatorNode🪅"

class StableAnimatorModels:

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
            },
        }

    RETURN_TYPES = ("STABLEANIMATORMODELS",)
    RETURN_NAMES = ("stable_animator_models",)
    FUNCTION = "load_models"
    CATEGORY = GLOBAL_CATEGORY
    # OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.models_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # 模型路径配置
        self.root_dir = folder_paths.get_folder_paths("stable_animator")[0]
        # os.path.join(, "stable_animator")
        self.model_config = {
            "pretrained_model": "stable-video-diffusion-img2vid-xt",
            "pose_net": "Animation/pose_net.pth",
            "face_encoder": "Animation/face_encoder.pth",
            "unet": "Animation/unet.pth"
        }

    def load_models(self):
        # 初始化模型组件
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            os.path.join(self.root_dir, self.model_config["pretrained_model"]),
            subfolder="feature_extractor"
        )
        
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            os.path.join(self.root_dir, self.model_config["pretrained_model"]),
            subfolder="scheduler"
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(self.root_dir, self.model_config["pretrained_model"]),
            subfolder="image_encoder"
        )

        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            os.path.join(self.root_dir, self.model_config["pretrained_model"]),
            subfolder="vae"
        )

        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(
            os.path.join(self.root_dir, self.model_config["pretrained_model"]),
            subfolder="unet",
            low_cpu_mem_usage=True
        )

        # 加载自定义组件
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])
        self.face_encoder = FusionFaceId(
            cross_attention_dim=1024,
            id_embeddings_dim=512,
            clip_embeddings_dim=1024,
            num_tokens=4
        )
        
        # 加载预训练权重
        self._load_weights(
            self.pose_net,
            os.path.join(self.root_dir, self.model_config["pose_net"])
        )
        self._load_weights(
            self.face_encoder,
            os.path.join(self.root_dir, self.model_config["face_encoder"])
        )
        self._load_weights(
            self.unet,
            os.path.join(self.root_dir, self.model_config["unet"])
        )

        # 配置注意力处理器
        self._configure_attention_processors()
        
        # 冻结模型参数
        for model in [self.vae, self.image_encoder, self.unet, self.pose_net, self.face_encoder]:
            model.requires_grad_(False)
            model.to(device=self.device, dtype=self.dtype)

        self.face_model = FaceModel(self.root_dir)

        return self,

    def _configure_attention_processors(self):
        """配置注意力处理器（保持原始实现逻辑）"""
        lora_rank = 128
        attn_procs = {}
        unet = self.unet
        unet_svd = unet.state_dict()

        for name in unet.attn_processors.keys():
            if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                if cross_attention_dim is None:
                    # print(f"This is AnimationAttnProcessor: {name}")
                    attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                else:
                    # print(f"This is AnimationIDAttnProcessor: {name}")
                    layer_name = name.split(".processor")[0]
                    weights = {
                        "to_k_ip.weight": unet_svd[layer_name + ".to_k.weight"],
                        "to_v_ip.weight": unet_svd[layer_name + ".to_v.weight"],
                    }
                    attn_procs[name] = AnimationIDAttnNormalizedProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                    attn_procs[name].load_state_dict(weights, strict=False)
            elif "temporal_transformer_blocks" in name:
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                if cross_attention_dim is None:
                    attn_procs[name] = XFormersAttnProcessor()
                else:
                    attn_procs[name] = XFormersAttnProcessor()
        unet.set_attn_processor(attn_procs)

    def _load_weights(self, model, path):
        """加载权重辅助函数"""
        if path.endswith(".pth"):
            state_dict = torch.load(path, map_location="cpu",weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.to(device=self.device, dtype=self.dtype)
        else:
            raise ValueError(f"Invalid model weights: {path}")



class StableAnimatorNode:
    """
    StableAnimator视频生成节点
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": {
                "stable_animator_models":("STABLEANIMATORMODELS",),
                "reference_image": ("IMAGE",),
                "pose_images_dir": ("STRING",{"default":"",}),  # 输入为图像序列文件夹路径
                "format":(["512x512","576x1024"],{}),
                "tile_size":("INT",{"default":16,"min":16,"max":64,"step":16}),
                "frames_overlap":("INT",{"default":4,"min":4,"max":64,"step":4}),
                "decode_chunk_size":("INT",{"default":4,"min":4,"max":64,"step":4}),
                "num_inference_steps":("INT",{"default":25,"min":1,"max":64,"step":1}),
                "noise_aug_strength": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.01,
                    "max": 1,
                    "step": 0.01
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "num_inference_steps": ("INT", {
                    "default": 25,
                    "min": 10,
                    "max": 50,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32-1
                }),
            },
            "optional": {
                "fps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 60
                })
            }
        }

    FUNCTION = "generate"
    CATEGORY = GLOBAL_CATEGORY

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    # OUTPUT_IS_LIST = (True,)

    def generate(self, **kwargs):
        """执行视频生成"""

        # 准备输入参数
        width, height = (512,512) if kwargs["format"]=="512x512" else (576,1024)

        models = kwargs["stable_animator_models"]
        reference_image = tensor_to_pil(kwargs["reference_image"])
        reference_image_np = tensor_to_np(kwargs["reference_image"])
        pose_images_dir = kwargs["pose_images_dir"]
        pose_images =  load_images_from_folder(pose_images_dir, width=width, height=height)
        seed = kwargs["seed"] if kwargs["seed"] != -1 else torch.randint(0, 2**32-1, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        num_frames = len(pose_images)
        
        # 初始化pipeline
        pipeline = InferenceAnimationPipeline(
            vae=models.vae,
            image_encoder=models.image_encoder,
            unet=models.unet,
            scheduler=models.noise_scheduler,
            feature_extractor=models.feature_extractor,
            pose_net=models.pose_net,
            face_encoder=models.face_encoder,
        ).to(device=models.device, dtype=models.dtype)

        face_model = models.face_model
        face_model.face_helper.clean_all()
        validation_image_face_info = models.face_model.app.get(reference_image_np)
        if len(validation_image_face_info) > 0:
            validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
            validation_image_id_ante_embedding = validation_image_face_info['embedding']
        else:
            validation_image_id_ante_embedding = None

        if validation_image_id_ante_embedding is None:
            face_model.face_helper.read_image(reference_image_np)
            face_model.face_helper.get_face_landmarks_5(only_center_face=True)
            face_model.face_helper.align_warp_face()

            if len(face_model.face_helper.cropped_faces) == 0:
                validation_image_id_ante_embedding = np.zeros((512,))
            else:
                validation_image_align_face = face_model.face_helper.cropped_faces[0]
                print('fail to detect face using insightface, extract embedding on align face')
                validation_image_id_ante_embedding = face_model.handler_ante.get_feat(validation_image_align_face)
        
        # 运行推理
        video_frames = pipeline(
            image=reference_image,
            image_pose=pose_images,
            height=height,
            width=width,
            num_frames=num_frames,
            tile_size=kwargs["tile_size"],
            tile_overlap=kwargs["frames_overlap"],
            decode_chunk_size=kwargs["decode_chunk_size"],
            motion_bucket_id=127.,
            fps=7,
            min_guidance_scale=kwargs["guidance_scale"],
            max_guidance_scale=kwargs["guidance_scale"],
            noise_aug_strength=kwargs["noise_aug_strength"],
            num_inference_steps=kwargs["num_inference_steps"],
            generator=generator,
            output_type="pil",
            validation_image_id_ante_embedding=validation_image_id_ante_embedding,
        ).frames[0]

        # 转换为张量输出
        output_tensors = []
        np_frames=np.array(video_frames)
        print("********************")
        print(np_frames.shape)
        print(np_frames[0].shape)
        # for frame in video_frames:
        #     np_frames = np.array(frame)
            # output_tensors.append(np_to_tensor(np.array(frame)))

        output_root = folder_paths.get_output_directory()
        output_dir = os.path.join(output_root, "stable_animator_output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{time.time()}.mp4")
        save_frames_as_mp4(np_frames, output_file, fps=kwargs["fps"])
        return (output_file,)
        # return (output_tensors,)

NODE_CLASS_MAPPINGS = {
    "StableAnimatorNode": StableAnimatorNode,
    "StableAnimatorModels": StableAnimatorModels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAnimatorNode": "HJH-StableAnimator - Video Generation",
    "StableAnimatorModels":"HJH-StableAnimator - Load Models"
}