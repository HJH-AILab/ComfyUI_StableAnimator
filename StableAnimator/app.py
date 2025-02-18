import os
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

import sys

# 获取当前文件的目录（root/test）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 root/test 添加到 sys.path
sys.path.append(current_dir)

from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet import UNetSpatioTemporalConditionModel
from animation.pipelines.inference_pipeline_animation import InferenceAnimationPipeline
import random

import gradio as gr
import gc
from datetime import datetime
from pathlib import Path

root = "E:/ai/ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI_StableAnimator/StableAnimator"
pretrained_model_name_or_path = f"{root}/checkpoints/stable-video-diffusion-img2vid-xt"
revision = None
posenet_model_name_or_path = f"{root}/checkpoints/Animation/pose_net.pth"
face_encoder_model_name_or_path = f"{root}/checkpoints/Animation/face_encoder.pth"
unet_model_name_or_path = f"{root}/checkpoints/Animation/unet.pth"


def load_images_from_folder(folder, width, height):
    images = []
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        img = img.resize((width, height))
        images.append(img)

    return images


def save_frames_as_png(frames, output_path):
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    num_frames = len(pil_frames)
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        save_path = os.path.join(output_path, f'frame_{i}.png')
        pil_frame.save(save_path)


def save_frames_as_mp4(frames, output_mp4_path, fps):
    print("Starting saving the frames as mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'H264' for better quality
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    for frame in frames:
       frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       out.write(frame_bgr)
    out.release()


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=125,
                       loop=0)


def generate(
    image_input: str, 
    pose_input: str, 
    width: int, 
    height: int, 
    guidance_scale: float, 
    num_inference_steps: int, 
    fps: int, 
    frames_overlap: int, 
    tile_size: int, 
    noise_aug_strength: float, 
    decode_chunk_size: int,
    seed: int,
):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir = os.path.join(output_dir, timestamp)
    if seed == -1:
        seed = random.randint(1, 2**20 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device=device, dtype=dtype)

    validation_image_path = image_input
    validation_image = Image.open(image_input).convert('RGB')
    validation_control_images = load_images_from_folder(pose_input, width=width, height=height)

    num_frames = len(validation_control_images)
    face_model.face_helper.clean_all()
    validation_face = cv2.imread(validation_image_path)
    validation_image_bgr = cv2.cvtColor(validation_face, cv2.COLOR_RGB2BGR)
    validation_image_face_info = face_model.app.get(validation_image_bgr)
    if len(validation_image_face_info) > 0:
        validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        validation_image_id_ante_embedding = validation_image_face_info['embedding']
    else:
        validation_image_id_ante_embedding = None

    if validation_image_id_ante_embedding is None:
        face_model.face_helper.read_image(validation_image_bgr)
        face_model.face_helper.get_face_landmarks_5(only_center_face=True)
        face_model.face_helper.align_warp_face()

        if len(face_model.face_helper.cropped_faces) == 0:
            validation_image_id_ante_embedding = np.zeros((512,))
        else:
            validation_image_align_face = face_model.face_helper.cropped_faces[0]
            print('fail to detect face using insightface, extract embedding on align face')
            validation_image_id_ante_embedding = face_model.handler_ante.get_feat(validation_image_align_face)

    # generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    decode_chunk_size = decode_chunk_size
    video_frames = pipeline(
        image=validation_image,
        image_pose=validation_control_images,
        height=height,
        width=width,
        num_frames=num_frames,
        tile_size=tile_size,
        tile_overlap=frames_overlap,
        decode_chunk_size=decode_chunk_size,
        motion_bucket_id=127.,
        fps=7,
        min_guidance_scale=guidance_scale,
        max_guidance_scale=guidance_scale,
        noise_aug_strength=noise_aug_strength,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="pil",
        validation_image_id_ante_embedding=validation_image_id_ante_embedding,
    ).frames[0]

    out_file = os.path.join(
        output_dir,
        f"animation_video.mp4",
    )
    for i in range(num_frames):
        img = video_frames[i]
        video_frames[i] = np.array(img)

    png_out_file = os.path.join(output_dir, "animated_images")
    os.makedirs(png_out_file, exist_ok=True)

    save_frames_as_mp4(video_frames, out_file, fps)
    export_to_gif(video_frames, out_file, fps)
    save_frames_as_png(video_frames, png_out_file)

    seed_update = gr.update(visible=True, value=seed)
    
    return out_file, seed_update


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">StableAnimator</h2>
            </div>
            <div style="text-align: center;">
                <a href="https://github.com/Francis-Rings/StableAnimator">🌐 Github</a> |
                <a href="https://arxiv.org/abs/2411.17697">📜 arXiv </a>
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ This demo is for academic research and experiential use only.
            </div>
            """)
    with gr.Row():
        with gr.Column():
            with gr.Group():
                image_input = gr.Image(label="Reference Image", type="filepath")
                pose_input = gr.Textbox(label="Driven Poses", placeholder="Please enter your driven pose directory here.")
            with gr.Group():
                with gr.Row():
                    width = gr.Number(label="Width (supports only 512×512 and 576×1024)", value=512)
                    height = gr.Number(label="Height (supports only 512×512 and 576×1024)", value=512)
                with gr.Row():
                    guidance_scale = gr.Number(label="Guidance scale (recommended 3.0)", value=3.0, step=0.1, precision=1)
                    num_inference_steps = gr.Number(label="Inference steps (recommended 25)", value=20)
                with gr.Row():
                    fps = gr.Number(label="FPS", value=8)
                    frames_overlap = gr.Number(label="Overlap Frames (recommended 4)", value=4)
                with gr.Row():
                    tile_size = gr.Number(label="Tile Size (recommended 16)", value=16)
                    noise_aug_strength = gr.Number(label="Noise Augmentation Strength (recommended 0.02)", value=0.02, step=0.01, precision=2)
                with gr.Row():
                    decode_chunk_size = gr.Number(label="Decode Chunk Size (recommended 4 or 16)", value=4)
                    seed = gr.Number(label="Random Seed (Enter a positive number, -1 for random)", value=-1)
            generate_button = gr.Button("🎬 Generate The Video")
        with gr.Column():
            video_output = gr.Video(label="Generate The Video")
            with gr.Row():
                seed_text = gr.Number(label="Video Generation Seed", visible=False, interactive=False)
    gr.Examples([
        ["inference/case-1/reference.png","inference/case-1/poses",512,512],
        ["inference/case-2/reference.png","inference/case-2/poses",512,512],
        ["inference/case-3/reference.png","inference/case-3/poses",512,512],
        ["inference/case-4/reference.png","inference/case-4/poses",512,512],
        ["inference/case-5/reference.png","inference/case-5/poses",576,1024],
    ], inputs=[image_input, pose_input, width, height])


    generate_button.click(
        generate,
        inputs=[image_input, pose_input, width, height, guidance_scale, num_inference_steps, fps, frames_overlap, tile_size, noise_aug_strength, decode_chunk_size, seed],
        outputs=[video_output, seed_text],
    )


if __name__ == "__main__":
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path, subfolder="feature_extractor", revision=revision)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="image_encoder", revision=revision)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        # clip_embeddings_dim=image_encoder.config.hidden_size,
        clip_embeddings_dim=1024,
        num_tokens=4, )
    face_model = FaceModel()

    lora_rank = 128
    attn_procs = {}
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

    # resume the previous checkpoint
    if posenet_model_name_or_path is not None and face_encoder_model_name_or_path is not None and unet_model_name_or_path is not None:
        print("Loading existing posenet weights, face_encoder weights and unet weights.")
        if posenet_model_name_or_path.endswith(".pth"):
            pose_net_state_dict = torch.load(posenet_model_name_or_path, map_location="cpu")
            pose_net.load_state_dict(pose_net_state_dict, strict=True)
        else:
            print("posenet weights loading fail")
            print(1/0)
        if face_encoder_model_name_or_path.endswith(".pth"):
            face_encoder_state_dict = torch.load(face_encoder_model_name_or_path, map_location="cpu")
            face_encoder.load_state_dict(face_encoder_state_dict, strict=True)
        else:
            print("face_encoder weights loading fail")
            print(1/0)
        if unet_model_name_or_path.endswith(".pth"):
            unet_state_dict = torch.load(unet_model_name_or_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=True)
        else:
            print("unet weights loading fail")
            print(1/0)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32mCUDA version：{torch.version.cuda}\033[0m')
    print(f'\033[32mPytorch version：{torch.__version__}\033[0m')
    print(f'\033[32mGPU Type：{torch.cuda.get_device_name()}\033[0m')
    print(f'\033[32mGPU Memory：{total_vram_in_gb:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32mSupports BF16, use BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32mBF16 is not supported, use FP16. The 5B model is not recommended\033[0m')
        dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo.queue()
    demo.launch(inbrowser=True)
