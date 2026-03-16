# ComfyUI-OmniLottie

A fully featured suite of custom nodes for **OmniLottie**, an end-to-end multimodal Lottie generator. Generate high-quality vector animations from Text, Image, and Video inputs directly in ComfyUI.

Optimized for **Intel Arc A770 (16GB)** and XPU hardware.

## Features

- **Multimodal Generation**: Text-to-Lottie, Image-to-Lottie, and Video-to-Lottie.
- **Intel Arc Optimization**: Native XPU support, aggressive VRAM management (15.2GB model on 16GB VRAM), and optional IPEX acceleration.
- **Interactive Visualizer**: A custom web player to see and play your animations within the ComfyUI interface.
- **Professional Toolkit**: Composition merging, canvas resizing, color skinning, and sprite sheet packing.
- **Game Dev Ready**: Extract collision boxes and export frame-accurate sprite sheets for Unity, Godot, and more.
- **Automated Setup**: One-click model downloading from Hugging Face and automatic script fetching from GitHub.

## Installation

1. Clone this repository to `ComfyUI/custom_nodes/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

## Credits

- **Author**: [aequanima](https://github.com/aequanima)
- **Model**: [OpenVGLab OmniLottie](https://github.com/OpenVGLab/OmniLottie)

## License

MIT
