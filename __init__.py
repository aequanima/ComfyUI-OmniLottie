__author__ = "aequanima"
__github__ = "https://github.com/aequanima/ComfyUI-OmniLottie"
__version__ = "2.0.0"

from .nodes import (
    OmniLottieModelManager,
    OmniLottieGenerator,
    OmniLottiePromptCrafter,
    OmniLottieEditor,
    OmniLottieExporter,
    OmniLottieUtilityHub,
    OmniLottieVisualizer,
    ImageToPalette,
    SaveLottie
)

NODE_CLASS_MAPPINGS = {
    "OmniLottieModelManager": OmniLottieModelManager,
    "OmniLottieGenerator": OmniLottieGenerator,
    "OmniLottiePromptCrafter": OmniLottiePromptCrafter,
    "OmniLottieEditor": OmniLottieEditor,
    "OmniLottieExporter": OmniLottieExporter,
    "OmniLottieUtilityHub": OmniLottieUtilityHub,
    "OmniLottieVisualizer": OmniLottieVisualizer,
    "ImageToPalette": ImageToPalette,
    "SaveLottie": SaveLottie
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniLottieModelManager": "OmniLottie Model Manager",
    "OmniLottieGenerator": "OmniLottie Generator",
    "OmniLottiePromptCrafter": "OmniLottie Prompt Crafter",
    "OmniLottieEditor": "OmniLottie Editor",
    "OmniLottieExporter": "OmniLottie Exporter",
    "OmniLottieUtilityHub": "OmniLottie Utility Hub",
    "OmniLottieVisualizer": "OmniLottie Visualizer",
    "ImageToPalette": "Image to Palette",
    "SaveLottie": "Save Lottie JSON"
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
