[English](https://github.com/HJH-AILab/ComfyUI_StableAnimator) | 中文版

# ComfyUI_StableAnimator
StableAnimator 的 ComfyUI 自定义节点.
原项目请访问 https://github.com/Francis-Rings/StableAnimator

# 功能
1. 独立了模型加载节点, 符合 comfyui 缓存机制
2. 制作了StableAnimator的从视频帧导出骨骼图节点, 你也可以使用comfyui_controlnet_aux的DWPose Estimator来生成骨骼图
3. 制作了从目录读取骨骼图的节点.
4. 现在节点已经可以正常使用了.
5. 在根目录下预设了StableAnimator目录,并且增加了__init__.py文件,!!不要移除, 以保证子包正确引用.
6. 工作流示例稍后提供...

# 建议
1. 建议使用ComfyUI-VideoHelperSuite来导出是视频帧和合成视频, 参考:https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
2. 建议在个人设备(显存较小的设备)上, 分别运行导出骨骼图和生成动作视频的流程.

# 安装
1. 拉取本项目到 ComfyUI/custom_nodes
2. 拉取 StableAnimator 到 ComfyUI/custom_nodes/ComfyUI_StableAnimator/StableAnimator
3. 按照 StableAnimator 项目README 步骤安装依赖，参考：https://github.com/Francis-Rings/StableAnimator

# Reward
Our team's reward code:

<img src="images/20250219-203952.png" alt="Out team's reward code" width="300">
