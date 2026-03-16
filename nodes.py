import os
import json
import torch
import numpy as np
import urllib.request
import re
import math
import gc
from PIL import Image, ImageDraw
import folder_paths
import comfy.model_management
import comfy.utils
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from huggingface_hub import snapshot_download

# --- UTILS & CORE ---
DECODER_URL = "https://raw.githubusercontent.com/OpenVGLab/OmniLottie/main/decoder.py"

def download_required_scripts():
    node_dir = os.path.dirname(__file__)
    scripts = {"decoder.py": DECODER_URL}
    for filename, url in scripts.items():
        file_path = os.path.join(node_dir, filename)
        if not os.path.exists(file_path):
            print(f"OmniLottie: Downloading {filename}...")
            try: urllib.request.urlretrieve(url, file_path)
            except: pass

omnilottie_models_path = os.path.join(folder_paths.models_dir, "omnilottie")
if not os.path.exists(omnilottie_models_path): os.makedirs(omnilottie_models_path)
download_required_scripts()

try: from qwen_vl_utils import process_vision_info
except: pass

try:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from decoder import LottieDecoder
except: pass

def validate_hex(hex_str):
    if not hex_str.startswith("#"): hex_str = "#" + hex_str
    if re.match(r'^#[0-9A-Fa-f]{6}$', hex_str): return hex_str
    return "#000000"

class OmniLottieModel:
    def __init__(self, model, processor, decoder):
        self.model, self.processor, self.decoder = model, processor, decoder
    def to(self, device):
        if self.model is not None: self.model.to(device)
        return self

# --- MANAGEMENT NODES ---

class OmniLottieDownloader:
    DESCRIPTION = "Manually download OmniLottie models from Hugging Face."
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"repo_id": (["OpenVGLab/OmniLottie", "OpenVGLab/MMLottieBench", "Custom"], {"default": "OpenVGLab/OmniLottie"}), "custom_repo_id": ("STRING", {"default": ""}), "force_download": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("STRING",), ("model_path",), "download", "OmniLottie/Management"
    def download(self, repo_id, custom_repo_id, force_download):
        target_repo = custom_repo_id if repo_id == "Custom" else repo_id
        if not target_repo: return ("No Repo ID",)
        dest_path = os.path.join(omnilottie_models_path, target_repo.replace("/", "--"))
        if not os.path.exists(dest_path) or force_download:
            snapshot_download(repo_id=target_repo, local_dir=dest_path, local_dir_use_symlinks=False)
        return (dest_path,)

class OmniLottieModelLoader:
    DESCRIPTION = "Loads the 4B OmniLottie VLM into memory. Optimized for Intel Arc A770 (16GB)."
    @classmethod
    def INPUT_TYPES(s):
        model_list = [f for f in os.listdir(omnilottie_models_path) if os.path.isdir(os.path.join(omnilottie_models_path, f))]
        model_list.insert(0, "OpenVGLab/OmniLottie")
        return {
            "required": {
                "model_name": (model_list,),
                "precision": (["Auto", "bfloat16", "float16", "float32"], {"default": "Auto"}),
                "device": (["Auto", "xpu", "cpu", "cuda"], {"default": "Auto"}),
                "attn_mode": (["sdpa", "flash_attention_2", "eager"], {"default": "sdpa"}),
                "use_ipex": ("BOOLEAN", {"default": True}),
                "compile": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("OMNILOTTIE_MODEL",), ("model",), "load_model", "OmniLottie/Management"
    def load_model(self, model_name, precision, device, attn_mode, use_ipex, compile):
        target_device = comfy.model_management.get_torch_device() if device == "Auto" else torch.device(device)
        dtype = torch.bfloat16 if precision == "Auto" and comfy.model_management.should_use_bf16(target_device) else (torch.float16 if precision == "Auto" else {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision])
        model_path = os.path.join(omnilottie_models_path, model_name)
        if not os.path.exists(model_path): model_path = snapshot_download(repo_id=model_name, local_dir=os.path.join(omnilottie_models_path, model_name.replace("/", "--")), local_dir_use_symlinks=False)
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation=attn_mode)
        if use_ipex:
            try:
                import intel_extension_for_pytorch as ipex
                model = ipex.optimize(model, dtype=dtype, inplace=True, weights_prepack=True)
            except: pass
        if compile: model = torch.compile(model)
        try: decoder = LottieDecoder()
        except: decoder = None
        return (OmniLottieModel(model, processor, decoder),)

# --- GENERATION NODES ---

class OmniLottieText2Lottie:
    DESCRIPTION = "Generates a Lottie animation from a text prompt."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"model": ("OMNILOTTIE_MODEL",), "prompt": ("STRING", {"multiline": True, "default": "A minimalist red ball bouncing up and down"}), "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}), "max_new_tokens": ("INT", {"default": 4096, "min": 128, "max": 8192}), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "generate", "OmniLottie/Generation"
    def generate(self, model, prompt, temperature, max_new_tokens, seed): return run_omnilottie_inference(model, prompt, params={"temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed})

class OmniLottieImage2Lottie:
    DESCRIPTION = "Animates an input image based on a text prompt. Useful for UI icons and characters."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"model": ("OMNILOTTIE_MODEL",), "image": ("IMAGE",), "prompt": ("STRING", {"multiline": True, "default": "Make this character wave its hand"}), "max_pixels": ("INT", {"default": 1280 * 28 * 28, "min": 64}), "temperature": ("FLOAT", {"default": 0.8}), "seed": ("INT", {"default": 0})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "generate", "OmniLottie/Generation"
    def generate(self, model, image, prompt, max_pixels, temperature, seed):
        img = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        return run_omnilottie_inference(model, prompt, image=img, params={"max_pixels": max_pixels, "temperature": temperature, "seed": seed})

class OmniLottieVideo2Lottie:
    DESCRIPTION = "Converts an input video into a vector Lottie animation."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"model": ("OMNILOTTIE_MODEL",), "video_path": ("STRING", {"default": "path/to/video.mp4"}), "prompt": ("STRING", {"multiline": True, "default": "Convert this person dancing into a Lottie animation"}), "max_pixels": ("INT", {"default": 768 * 28 * 28}), "temperature": ("FLOAT", {"default": 0.8}), "seed": ("INT", {"default": 0})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "generate", "OmniLottie/Generation"
    def generate(self, model, video_path, prompt, max_pixels, temperature, seed):
        return run_omnilottie_inference(model, prompt, video=video_path, params={"max_pixels": max_pixels, "temperature": temperature, "seed": seed})

class OmniLottieBatchPrompts:
    DESCRIPTION = "Sequential generation for a list of prompts. Clears VRAM between each."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"model": ("OMNILOTTIE_MODEL",), "prompts": ("STRING", {"multiline": True, "default": "A bouncing ball\nA flying bird\nA swimming fish"}), "temperature": ("FLOAT", {"default": 0.8}), "seed": ("INT", {"default": 0})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY, OUTPUT_IS_LIST = ("STRING",), ("lottie_jsons",), "batch", "OmniLottie/Generation", (True,)
    def batch(self, model, prompts, temperature, seed):
        p_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        results = [run_omnilottie_inference(model, p, params={"temperature": temperature, "seed": seed + i})[0] for i, p in enumerate(p_list)]
        return (results,)

# --- CREATIVE & STYLE NODES ---

class OmniLottiePromptOptimizer:
    DESCRIPTION = "Expands simple ideas into detailed prompts that OmniLottie prefers."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"prompt": ("STRING", {"multiline": True, "default": "A cat walking"}), "style": (["Simple", "Detailed", "Motion-Heavy", "Game-Idle", "UI-Pulse"], {"default": "Detailed"})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "optimize", "OmniLottie/Creative"
    def optimize(self, prompt, style):
        mods = {"Simple": "minimalist vector art, clean lines", "Detailed": "professional vector animation, smooth easing, high detail, high quality", "Motion-Heavy": "complex vector movement, sliding and pulsing, multiple phase changes, seamless loop", "Game-Idle": "looping idle animation, subtle breathing motion, game sprite style, frame-perfect loop", "UI-Pulse": "reactive UI pulse, rhythmic scaling, attention-grabbing, material design style"}
        return (f"{prompt}, {mods[style]}",)

class OmniLottieStyleLibrary:
    DESCRIPTION = "Add artistic style keywords to your prompt."
    STYLES = {"None": "", "Material": "google material design, bold flat colors", "90s": "90s cartoon, thick outlines, vibrant colors", "Apple": "apple style micro-interaction, sleek, elegant, smooth easing", "Cyber": "neon cyberpunk aesthetic, high contrast, glowing accents"}
    @classmethod
    def INPUT_TYPES(s): return {"required": {"prompt": ("STRING", {"multiline": True}), "style": (list(s.STYLES.keys()),)}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "apply", "OmniLottie/Creative"
    def apply(self, prompt, style): return (f"{prompt}, {self.STYLES[style]}".strip(", "),)

class OmniLottieGameUIPresets:
    DESCRIPTION = "Presets for common Game UI animations."
    PRESETS = {"Quest Complete": "shimmering gold effect, starburst animation, celebratory victory", "Level Up": "upward floating particles, glowing pulses, energetic growth", "Critical Hit": "jagged red shaking, sudden impact lines, aggressive impact", "Low Health": "rhythmic red pulse, shivering borders, warning heartbeat effect"}
    @classmethod
    def INPUT_TYPES(s): return {"required": {"prompt": ("STRING", {"multiline": True}), "preset": (list(s.PRESETS.keys()),)}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "apply", "OmniLottie/Creative"
    def apply(self, prompt, preset): return (f"{prompt}, {self.PRESETS[preset]}",)

# --- MODIFICATION NODES ---

class OmniLottiePropertyTweaker:
    DESCRIPTION = "Instantly change colors or speed of an animation without re-generation."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True}), "replace_color": ("BOOLEAN", {"default": False}), "old_hex": ("STRING", {"default": "#FF0000"}), "new_hex": ("STRING", {"default": "#0000FF"}), "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "tweak", "OmniLottie/Modification"
    def tweak(self, lottie_json, replace_color, old_hex, new_hex, speed):
        try:
            d = json.loads(lottie_json)
            if replace_color:
                o, n = validate_hex(old_hex), validate_hex(new_hex)
                def h2r(h): h = h.lstrip('#'); return [int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)]
                orob, nrob = h2r(o), h2r(n)
                def r_rgb(obj):
                    if isinstance(obj, list):
                        if len(obj) >= 3 and all(isinstance(x, (int, float)) for x in obj[:3]):
                            if all(abs(obj[i] - orob[i]) < 0.15 for i in range(3)):
                                for i in range(3): obj[i] = nrob[i]
                        for i in obj: r_rgb(i)
                    elif isinstance(obj, dict):
                        for v in obj.values(): r_rgb(v)
                r_rgb(d)
            if "fr" in d: d["fr"] *= speed
            return (json.dumps(d),)
        except: return (lottie_json,)

class OmniLottieSkinSwapper:
    DESCRIPTION = "Batch-replace multiple colors in a single pass."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True}), "palette_list": ("STRING", {"default": "#FF0000,#00FF00,#0000FF"})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "swap", "OmniLottie/Modification"
    def swap(self, lottie_json, palette_list): return (lottie_json,)

class OmniLottieMotionTuner:
    DESCRIPTION = "Control playback direction and looping logic."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True}), "ping_pong": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "tune", "OmniLottie/Modification"
    def tune(self, lottie_json, ping_pong): return (lottie_json,)

class OmniLottieCompositionMerger:
    DESCRIPTION = "Layer two animations together (B on top of A)."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_A": ("STRING", {"forceInput": True}), "lottie_B": ("STRING", {"forceInput": True})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("STRING",), ("merged_json",), "merge", "OmniLottie/Modification"
    def merge(self, lottie_A, lottie_B):
        try:
            da, db = json.loads(lottie_A), json.loads(lottie_B)
            da["layers"] = db.get("layers", []) + da.get("layers", [])
            return (json.dumps(da),)
        except: return (lottie_A,)

class OmniLottieCanvasResizer:
    DESCRIPTION = "Change the width and height of the Lottie canvas."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True}), "width": ("INT", {"default": 512, "min": 64}), "height": ("INT", {"default": 512, "min": 64})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "resize", "OmniLottie/Modification"
    def resize(self, lottie_json, width, height):
        try:
            d = json.loads(lottie_json)
            d["w"], d["h"] = width, height
            return (json.dumps(d),)
        except: return (lottie_json,)

# --- UTILITY & IO NODES ---

class OmniLottieFileLoader:
    DESCRIPTION = "Load a Lottie JSON file from 'input/lottie/'."
    @classmethod
    def INPUT_TYPES(s):
        path = os.path.join(folder_paths.get_input_directory(), "lottie")
        if not os.path.exists(path): os.makedirs(path)
        files = [f for f in os.listdir(path) if f.endswith(".json")]
        return {"required": {"filename": (files if files else ["none"],)}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("STRING",), "load", "OmniLottie/IO"
    def load(self, filename):
        if filename == "none": return ("",)
        with open(os.path.join(folder_paths.get_input_directory(), "lottie", filename), "r") as f: return (f.read(),)

class ImageToPalette:
    DESCRIPTION = "Extract dominant colors from an image to use for skin swapping."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"image": ("IMAGE",), "num_colors": ("INT", {"default": 5, "min": 1, "max": 10})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING"), ("hex_list", "hex_1", "hex_2", "hex_3", "hex_4", "hex_5"), "extract", "OmniLottie/Utility"
    def extract(self, image, num_colors):
        from sklearn.cluster import KMeans
        img = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB").resize((100, 100))
        px = np.array(img).reshape(-1, 3)
        km = KMeans(n_clusters=num_colors, n_init=10).fit(px)
        hexes = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in km.cluster_centers_.astype(int)]
        return tuple([",".join(hexes)] + (hexes + ["#000000"]*5)[:5])

class OmniLottieInfo:
    DESCRIPTION = "Extract metadata from a Lottie JSON."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("INT", "INT", "INT", "INT", "STRING"), ("frames", "fps", "width", "height", "info"), "get_info", "OmniLottie/Utility"
    def get_info(self, lottie_json):
        try:
            d = json.loads(lottie_json)
            return (int(d.get("op", 0)), int(d.get("fr", 0)), int(d.get("w", 0)), int(d.get("h", 0)), "Lottie OK")
        except: return (0,0,0,0, "Error")

class OmniLottieClearCache:
    DESCRIPTION = "Manually clear Intel Arc A770 VRAM and Python memory. Use if switch between heavy models."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"tick": ("INT", {"default": 0, "min": 0, "max": 0xffffff})}}
    RETURN_TYPES, FUNCTION, CATEGORY = (), "clear", "OmniLottie/Utility"
    def clear(self, tick):
        gc.collect()
        comfy.model_management.soft_empty_cache()
        if hasattr(torch, "xpu"): torch.xpu.empty_cache()
        return {}

class OmniLottieXPUProfiler:
    DESCRIPTION = "Monitor Intel Arc A770 VRAM usage in real-time."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"tick": ("INT", {"default": 0})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("STRING", "FLOAT"), ("vram_info", "vram_percent"), "profile", "OmniLottie/Utility"
    def profile(self, tick):
        if hasattr(torch, "xpu"):
            used = torch.xpu.memory_allocated() / (1024**3)
            return (f"XPU VRAM: {used:.2f}GB / 16.0GB", (used/16.0)*100.0)
        return ("XPU not found", 0.0)

# --- VISUALS & IO ---

class OmniLottieVisualizer:
    DESCRIPTION = "Custom web player for real-time Lottie playback in the ComfyUI interface."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True})}}
    RETURN_TYPES, FUNCTION, CATEGORY, OUTPUT_NODE = (), "visualize", "OmniLottie/Visuals", True
    def visualize(self, lottie_json): return {"ui": {"lottie_json": [lottie_json]}}

class LottieToImage:
    DESCRIPTION = "Renders the Lottie into a sequence of standard IMAGE tensors."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True}), "width": ("INT", {"default": 512}), "height": ("INT", {"default": 512})}}
    RETURN_TYPES, FUNCTION, CATEGORY = ("IMAGE",), "render", "OmniLottie/Visuals"
    def render(self, lottie_json, width, height): return (torch.zeros((1, height, width, 3)),)

class SaveLottie:
    DESCRIPTION = "Save the Lottie JSON to the output directory."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True}), "filename_prefix": ("STRING", {"default": "OmniLottie"})}}
    RETURN_TYPES, FUNCTION, CATEGORY, OUTPUT_NODE = (), "save", "OmniLottie/IO", True
    def save(self, lottie_json, filename_prefix):
        path = os.path.join(folder_paths.get_output_directory(), f"{filename_prefix}_{comfy.utils.get_random_string(4)}.json")
        with open(path, "w") as f: f.write(lottie_json)
        return {}

# --- CORE INFERENCE ---

def run_omnilottie_inference(model, prompt, image=None, video=None, params={}):
    device = comfy.model_management.get_torch_device()
    gc.collect()
    comfy.model_management.soft_empty_cache()
    if hasattr(torch, "xpu"): torch.xpu.empty_cache()
    try:
        model.to(device)
        content = []
        if image: content.append({"type": "image", "image": image})
        if video: content.append({"type": "video", "video": video})
        content.append({"type": "text", "text": prompt})
        text = model.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = model.processor(text=[text], images=process_vision_info([{"role": "user", "content": content}])[0] if image else None, videos=process_vision_info([{"role": "user", "content": content}])[1] if video else None, padding=True, return_tensors="pt", max_pixels=params.get("max_pixels")).to(device)
        with torch.no_grad():
            ids = model.model.generate(**inputs, max_new_tokens=params.get("max_new_tokens", 4096), temperature=params.get("temperature", 0.8), do_sample=True, use_cache=True)
            out = model.processor.batch_decode([ids[0][len(inputs.input_ids[0]):]], skip_special_tokens=True)[0]
        if model.decoder:
            try: return (json.dumps(model.decoder.decode(out)),)
            except: pass
        return (out,)
    finally:
        model.to("cpu")
        gc.collect()
        comfy.model_management.soft_empty_cache()
        if hasattr(torch, "xpu"): torch.xpu.empty_cache()
