English | [中文版](https://github.com/HJH-AILab/ComfyUI_StableAnimator/blob/main/README.cn.md)

# ComfyUI_StableAnimator
Custom nodes for ComfyUI of StableAnimator.

Visit the original project at https://github.com/Francis-Rings/StableAnimator.

## Features
1. The model loading node has been made independent, which complies with the ComfyUI caching mechanism.
2. A node for exporting bone maps from video frames of StableAnimator has been created. You can also use the DWPose Estimator of comfyui_controlnet_aux to generate bone maps.
3. A node for reading bone maps from a directory has been created.
4. The nodes can now be used normally.
5. A StableAnimator directory is preset in the root directory, and an __init__.py file has been added. **Do not remove it** to ensure the correct reference of sub-packages.
6. Workflow examples will be provided later...

## Suggestions
1. It is recommended to use ComfyUI - VideoHelperSuite to export video frames and synthesize videos. Refer to: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.
2. On personal devices (devices with smaller video memory), it is recommended to run the processes of exporting bone maps and generating action videos separately.

## Installation
1. Pull this project to ComfyUI/custom_nodes.
2. Pull StableAnimator to ComfyUI/custom_nodes/ComfyUI_StableAnimator/StableAnimator.
3. Install the dependencies according to the steps in the README of the StableAnimator project. Refer to: https://github.com/Francis-Rings/StableAnimator.

## Reward
Our team's reward code:

<img src="images/20250219-203952.png" alt="Out team's reward code" width="300">
