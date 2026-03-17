# ComfyUI-OmniLottie (v2.0)

A professional, high-performance suite of ComfyUI custom nodes for **OmniLottie**, the state-of-the-art multimodal vector animation generator. 

Generate, edit, and export Lottie animations from Text, Image, and Video inputs with a streamlined, user-centered design optimized for the **Intel Arc A770 (16GB)**.

## 🚀 Key Features

- **Consolidated Hub Architecture**: 28+ specialized features merged into 10 powerful, intuitive nodes.
- **Extreme VRAM Optimization**: Features 8-bit and 4-bit quantization (via `bitsandbytes`) to drastically reduce the 15.2GB model footprint, perfect for Intel Arc A770.
- **ComfyUI Native Integration**: Uses `comfy.model_patcher` for intelligent, native XPU/CPU memory offloading alongside other heavy models like FLUX.
- **Web & App UX Pipeline**: Features a Bodymovin JSON compressor (rounds floats to save KB), an Auto-Dark Mode inverter, and specialized Micro-Interaction Prompt Presets.
- **Frontend Code Generator**: Outputs ready-to-use React (`.jsx`), Vue 3, or Vanilla HTML wrapper code for your Lottie animations.
- **Custom UI Theme & UX**: Features a custom "Solarized" UI theme, dynamic auto-hiding widgets to reduce clutter, and a live VRAM Heatmap.
- **Universal Hardware Compatibility**: Heavily optimized for **Intel Arc A770 (16GB)** with IPEX and SDPA attention, but includes graceful fallbacks for **CUDA (Nvidia)** and **MPS (Apple Silicon)**.
- **Multimodal Hub**: Seamlessly switch between Text-to-Lottie, Image-to-Lottie, and Video-to-Lottie in a single node, now with advanced generation parameters (`top_k`, `top_p`, `repetition_penalty`).
- **Instant Editor**: Tweak colors, compress JSON, and merge layers instantly without re-running the AI model.
- **Quick Export Visualizer**: Interactive web player features "One-Click" buttons to instantly Copy Code or Download JSON directly from the ComfyUI canvas.
- **Professional Export**: Export to Image Sequences, Masks, MP4 Video, SVG, or Game Engine SpriteSheets.

## 🛠️ The Node Suite

1. **Model Manager**: Handles automatic downloading and advanced hardware toggles.
2. **Generator**: The multimodal engine for all generation and batching tasks.
3. **Prompt Crafter**: Refines simple ideas into motion-optimized VLM instructions (includes Web UI and Game Dev presets).
4. **Editor**: Instant property modification, JSON compression, and Dark Mode auto-inversion.
5. **Exporter**: The rendering hub for all vector and raster formats.
6. **Frontend Exporter**: Generates React/Vue/HTML boilerplate code for your JSON.
7. **Utility Hub**: Hardware profiling, cache clearing, and local file loading.
8. **Visualizer**: Real-time interactive web player for instant inspection.
9. **Image to Palette**: Extract dominant colors to sync animations with source art.
10. **Save Lottie**: Standardized JSON output node.

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

