[English](https://github.com/HJH-AILab/ComfyUI_StableAnimator) | 中文版

## ComfyUI_StableAnimator
StableAnimator 的 ComfyUI 自定义节点.
原项目请访问 https://github.com/Francis-Rings/StableAnimator

## 功能
1. 独立了模型加载节点, 符合 comfyui 缓存机制
2. 制作了StableAnimator的从视频帧导出骨骼图节点, 你也可以使用comfyui_controlnet_aux的DWPose Estimator来生成骨骼图
3. 制作了从目录读取骨骼图的节点.
4. 现在节点已经可以正常使用了.

## Workflow
工作流文件在<a href="workflow/example.json">这里</a>
<br>
<img src="workflow/example" alt="a example workflow" width="100%">

## 建议
1. 建议使用ComfyUI-VideoHelperSuite来导出是视频帧和合成视频, 参考:https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
2. 建议在个人设备(显存较小的设备)上, 分别运行导出骨骼图和生成动作视频的流程.

## 安装
1. 拉取本项目到 ComfyUI/custom_nodes
2. 拉取 StableAnimator 到 ComfyUI/custom_nodes/ComfyUI_StableAnimator/StableAnimator
3. 按照 StableAnimator 项目README 步骤安装依赖，参考：https://github.com/Francis-Rings/StableAnimator
4. 模型下载到本地后，必须按一下目录结构部署。（正在做不同模块的拆解）

## 模型本地目录
```
stable_animator
        /Animation
                face_encoder.pth
                pose_net.pth
                unet.pth
        /antelopev2
                1k3d68.onnx
                2d106det.onnx
                genderage.onnx
                glintr100.onnx
                scrfd_10g_bnkps.onnx
        /DWPose(使用Contrelnet的DWPose预处理器节点可以不放这个)
                dw-ll_ucoco_384.onnx
                yolox_l.onnx
        /stable-video-diffusion-img2vid-xt
                svd_xt_image_decoder.safetensors
                svd_xt.safetensors
                model_index.json
                /feature_extractor
                        preprocessor_config.json
                /image_encoder
                        config.json
                        model.fp16.safetensors
                        model.safetensors
                /scheduler
                        scheduler_config.json
                /unet
                        config.json
                        diffusion_pytorch_model.fp16.safetensors
                        diffusion_pytorch_model.safetensors
                /vae
                        config.json
                        diffusion_pytorch_model.fp16.safetensors
                        diffusion_pytorch_model.safetensors
```

# Reward
Our team's reward code:

<img src="images/20250219-203952.png" alt="Out team's reward code" width="300">
