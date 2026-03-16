# ComfyUI-OmniLottie (v2.0)

A professional, high-performance suite of ComfyUI custom nodes for **OmniLottie**, the state-of-the-art multimodal vector animation generator. 

Generate, edit, and export Lottie animations from Text, Image, and Video inputs with a streamlined, user-centered design optimized for the **Intel Arc A770 (16GB)**.

## 🚀 Key Features

- **Consolidated Hub Architecture**: 28+ specialized features merged into 9 powerful, intuitive nodes.
- **Intel Arc Optimization**: Native XPU support, `bfloat16` precision, IPEX layout optimizations, and `torch.compile` support.
- **Aggressive VRAM Management**: Surgical memory offloading ensures the 15.2GB model runs smoothly on 16GB hardware.
- **Multimodal Hub**: Seamlessly switch between Text-to-Lottie, Image-to-Lottie, and Video-to-Lottie in a single node.
- **Instant Editor**: Tweak colors, speed, and canvas size or merge layers instantly without re-running the AI model.
- **Professional Export**: Export to Image Sequences, Masks, MP4 Video, SVG, or Game Engine SpriteSheets.
- **Creative Suite**: Built-in Prompt Optimizers, Style Libraries, and Game UI presets.

## 🛠️ The Node Suite

1. **Model Manager**: Handles automatic downloading and advanced hardware toggles.
2. **Generator**: The multimodal engine for all generation and batching tasks.
3. **Prompt Crafter**: Refines simple ideas into motion-optimized VLM instructions.
4. **Editor**: Instant property modification and layer composition.
5. **Exporter**: The rendering hub for all vector and raster formats.
6. **Utility Hub**: Hardware profiling, cache clearing, and local file loading.
7. **Visualizer**: Real-time interactive web player for instant inspection.
8. **Image to Palette**: Extract dominant colors to sync animations with source art.
9. **Save Lottie**: Standardized JSON output node.

## 📦 Installation

1. Clone this repository to `ComfyUI/custom_nodes/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

## 👥 Credits

- **Author**: [aequanima](https://github.com/aequanima)
- **Model**: [OpenVGLab OmniLottie](https://github.com/OpenVGLab/OmniLottie)

## License

MIT
