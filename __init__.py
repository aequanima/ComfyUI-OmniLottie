__author__ = "aequanima"
__github__ = "https://github.com/aequanima/ComfyUI-OmniLottie"
__version__ = "1.0.0"

from .nodes import (
    OmniLottieDownloader,
    OmniLottieModelLoader,
    OmniLottieText2Lottie,
    OmniLottieImage2Lottie,
    OmniLottieVideo2Lottie,
    OmniLottieBatchPrompts,
    OmniLottiePromptOptimizer,
    OmniLottieStyleLibrary,
    OmniLottieGameUIPresets,
    ImageToPalette,
    OmniLottiePropertyTweaker,
    OmniLottieSkinSwapper,
    OmniLottieMotionTuner,
    OmniLottieCompositionMerger,
    OmniLottieCanvasResizer,
    OmniLottieFileLoader,
    OmniLottieInfo,
    OmniLottieClearCache,
    OmniLottieCollisionAnalyzer,
    OmniLottieVisualizer,
    OmniLottieJSONPreview,
    LottieToImage,
    LottieToMask,
    LottieToVideo,
    LottieToSVG,
    OmniLottieSpriteSheetPacker,
    OmniLottieXPUProfiler,
    SaveLottie
)

NODE_CLASS_MAPPINGS = {
    # Management
    "OmniLottieDownloader": OmniLottieDownloader,
    "OmniLottieModelLoader": OmniLottieModelLoader,
    
    # Generation
    "OmniLottieText2Lottie": OmniLottieText2Lottie,
    "OmniLottieImage2Lottie": OmniLottieImage2Lottie,
    "OmniLottieVideo2Lottie": OmniLottieVideo2Lottie,
    "OmniLottieBatchPrompts": OmniLottieBatchPrompts,
    
    # Creative
    "OmniLottiePromptOptimizer": OmniLottiePromptOptimizer,
    "OmniLottieStyleLibrary": OmniLottieStyleLibrary,
    "OmniLottieGameUIPresets": OmniLottieGameUIPresets,
    "ImageToPalette": ImageToPalette,
    
    # Modification
    "OmniLottiePropertyTweaker": OmniLottiePropertyTweaker,
    "OmniLottieSkinSwapper": OmniLottieSkinSwapper,
    "OmniLottieMotionTuner": OmniLottieMotionTuner,
    "OmniLottieCompositionMerger": OmniLottieCompositionMerger,
    "OmniLottieCanvasResizer": OmniLottieCanvasResizer,
    
    # Utility & IO
    "OmniLottieFileLoader": OmniLottieFileLoader,
    "OmniLottieInfo": OmniLottieInfo,
    "OmniLottieClearCache": OmniLottieClearCache,
    "OmniLottieCollisionAnalyzer": OmniLottieCollisionAnalyzer,
    "OmniLottieXPUProfiler": OmniLottieXPUProfiler,
    "SaveLottie": SaveLottie,
    
    # Visuals
    "OmniLottieVisualizer": OmniLottieVisualizer,
    "OmniLottieJSONPreview": OmniLottieJSONPreview,
    "LottieToImage": LottieToImage,
    "LottieToMask": LottieToMask,
    "LottieToVideo": LottieToVideo,
    "LottieToSVG": LottieToSVG,
    
    # GameDev
    "OmniLottieSpriteSheetPacker": OmniLottieSpriteSheetPacker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniLottieDownloader": "OmniLottie Downloader",
    "OmniLottieModelLoader": "OmniLottie Model Loader",
    "OmniLottieText2Lottie": "Text to Lottie",
    "OmniLottieImage2Lottie": "Image to Lottie",
    "OmniLottieVideo2Lottie": "Video to Lottie",
    "OmniLottieBatchPrompts": "OmniLottie Batch Prompts",
    "OmniLottiePromptOptimizer": "Prompt Optimizer",
    "OmniLottieStyleLibrary": "Style Library",
    "OmniLottieGameUIPresets": "Game UI Presets",
    "ImageToPalette": "Image to Palette",
    "OmniLottiePropertyTweaker": "Property Tweaker",
    "OmniLottieSkinSwapper": "Skin Swapper",
    "OmniLottieMotionTuner": "Motion Tuner",
    "OmniLottieCompositionMerger": "Composition Merger",
    "OmniLottieCanvasResizer": "Canvas Resizer",
    "OmniLottieFileLoader": "Lottie File Loader",
    "OmniLottieInfo": "Lottie Info",
    "OmniLottieClearCache": "Clear OmniLottie Cache",
    "OmniLottieCollisionAnalyzer": "Collision Analyzer",
    "OmniLottieXPUProfiler": "XPU Profiler",
    "SaveLottie": "Save Lottie JSON",
    "OmniLottieVisualizer": "OmniLottie Visualizer",
    "OmniLottieJSONPreview": "JSON Preview",
    "LottieToImage": "Lottie to Image",
    "LottieToMask": "Lottie to Mask",
    "LottieToVideo": "Lottie to Video",
    "LottieToSVG": "Lottie to SVG",
    "OmniLottieSpriteSheetPacker": "SpriteSheet Packer"
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
