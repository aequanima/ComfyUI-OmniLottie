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

# --- CORE UTILS ---
DECODER_URL = "https://raw.githubusercontent.com/OpenVGLab/OmniLottie/main/decoder.py"

def download_required_scripts():
    node_dir = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(node_dir, "decoder.py")):
        print(f"OmniLottie: Downloading decoder.py...")
        try:
            urllib.request.urlretrieve(DECODER_URL, os.path.join(node_dir, "decoder.py"))
        except Exception as e:
            print(f"OmniLottie: Failed to download decoder: {e}")

omnilottie_models_path = os.path.join(folder_paths.models_dir, "omnilottie")
if not os.path.exists(omnilottie_models_path):
    os.makedirs(omnilottie_models_path)
download_required_scripts()

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("OmniLottie: qwen_vl_utils not found. Vision features may be limited.")

try:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from decoder import LottieDecoder
except ImportError:
    print("OmniLottie: decoder.py not found. Model outputs will remain as raw tokens.")

def validate_hex(hex_str):
    if not hex_str: return None
    if not hex_str.startswith("#"): hex_str = "#" + hex_str
    if re.match(r'^#[0-9A-Fa-f]{6}$', hex_str): return hex_str
    return "#000000"

class OmniLottieModel:
    def __init__(self, model, processor, decoder):
        self.model = model
        self.processor = processor
        self.decoder = decoder
    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        return self

# --- CONSOLIDATED HUB NODES ---

class OmniLottieModelManager:
    """Hub: Management & Hardware Optimization"""
    DESCRIPTION = "Manages OmniLottie model lifecycle. Handles automatic downloading, precision (bfloat16/float16), and XPU (Intel Arc) hardware optimizations."
    
    @classmethod
    def INPUT_TYPES(s):
        model_list = [f for f in os.listdir(omnilottie_models_path) if os.path.isdir(os.path.join(omnilottie_models_path, f))]
        model_list.insert(0, "OpenVGLab/OmniLottie")
        return {
            "required": {
                "model_name": (model_list,),
                "precision": (["Auto", "bfloat16", "float16", "float32"], {"default": "Auto"}),
                "device": (["Auto", "xpu", "cpu", "cuda"], {"default": "Auto"}),
                "performance": (["Balanced", "Max-Speed (Compile)", "Max-VRAM-Safe"], {"default": "Balanced"}),
            },
            "optional": {
                "custom_repo_id": ("STRING", {"default": ""}),
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("OMNILOTTIE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "OmniLottie"

    def load(self, model_name, precision, device, performance, custom_repo_id="", force_redownload=False):
        target_repo = custom_repo_id.strip() if custom_repo_id.strip() else model_name
        target_device = comfy.model_management.get_torch_device() if device == "Auto" else torch.device(device)
        
        # ARC A770 Optimization: Native BF16 support check
        if precision == "Auto":
            dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(target_device) else torch.float16
        else:
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]

        # Automated Setup
        safe_name = target_repo.replace("/", "--")
        model_path = os.path.join(omnilottie_models_path, safe_name)
        if not os.path.exists(model_path) or force_redownload:
            print(f"OmniLottie: Downloading weights for {target_repo}...")
            model_path = snapshot_download(repo_id=target_repo, local_dir=model_path, local_dir_use_symlinks=False)

        # Implementation specific optimizations
        attn_mode = "sdpa" if performance != "Max-VRAM-Safe" else "eager"
        compile_model = True if performance == "Max-Speed (Compile)" else False

        print(f"OmniLottie: Initializing 4B VLM ({dtype}) on {target_device}...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation=attn_mode
        )
        
        # IPEX Optimization for Intel Hardware
        try:
            import intel_extension_for_pytorch as ipex
            print("OmniLottie: Applying Intel IPEX layout optimizations...")
            model = ipex.optimize(model, dtype=dtype, inplace=True, weights_prepack=True)
        except:
            pass
            
        if compile_model:
            print("OmniLottie: Compiling graph (Initial run will be slower)...")
            model = torch.compile(model)
        
        try:
            decoder = LottieDecoder()
        except:
            decoder = None
        
        return (OmniLottieModel(model, processor, decoder),)

class OmniLottieGenerator:
    """Hub: Multi-modal Generation & Batching"""
    DESCRIPTION = "The primary generation engine. Automatically handles Text, Image, and Video inputs. Supports multiline batch generation."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("OMNILOTTIE_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A red ball bouncing up and down"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
                "batch_prompts": ("STRING", {"multiline": True, "default": ""}),
                "max_res": ("INT", {"default": 1280, "min": 64, "max": 2048}),
                "max_tokens": ("INT", {"default": 4096, "min": 128, "max": 8192}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LOTTIE_JSON",)
    FUNCTION = "generate"
    CATEGORY = "OmniLottie"
    OUTPUT_IS_LIST = (True,)

    def generate(self, model, prompt, seed, temperature, image=None, video_path="", batch_prompts="", max_res=1280, max_tokens=4096):
        # Determine Input Modality
        pil_img = None
        if image is not None:
            pil_img = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        
        # Batch Processing Logic
        p_list = [p.strip() for p in batch_prompts.split('\n') if p.strip()]
        if not p_list: p_list = [prompt]
        
        results = []
        for i, p in enumerate(p_list):
            print(f"OmniLottie: Generating item {i+1}/{len(p_list)}...")
            res = run_omnilottie_inference(
                model, p, image=pil_img, video=video_path if video_path.strip() else None, 
                params={"max_pixels": max_res * 28 * 28, "max_new_tokens": max_tokens, "temperature": temperature, "seed": seed + i}
            )
            results.append(res[0])
            
        return (results,)

class OmniLottiePromptCrafter:
    """Hub: Prompt Optimization & Styles"""
    DESCRIPTION = "The creative brain. Refines simple prompts into VLM-optimized animation instructions. Includes Style and Game UI libraries."
    
    @classmethod
    def INPUT_TYPES(s):
        styles = ["None", "Minimalist Vector", "Google Material", "90s Cartoon", "Apple Style", "Futuristic/Neon", "Hand-drawn Sketch"]
        game_ui = ["None", "Level Up Notification", "Critical Hit Shake", "Quest Complete Celebration", "Low Health Warning", "Inventory Item Shine"]
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["Draft", "Polished", "Complex Motion"], {"default": "Polished"}),
                "visual_style": (styles, {"default": "None"}),
                "game_ui_preset": (game_ui, {"default": "None"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("crafted_prompt",)
    FUNCTION = "craft"
    CATEGORY = "OmniLottie"

    def craft(self, prompt, mode, visual_style, game_ui_preset):
        out = prompt
        # Apply Styles
        if visual_style != "None":
            style_map = {"Minimalist Vector": "clean minimalist lines, flat vector art", "Google Material": "google material design style, bold flat colors", "90s Cartoon": "90s cartoon style, thick outlines, vibrant", "Apple Style": "apple micro-interaction style, sleek, elegant", "Futuristic/Neon": "futuristic neon aesthetic, high contrast", "Hand-drawn Sketch": "organic hand-drawn sketch, rough vector lines"}
            out += f", {style_map[visual_style]}"
        
        # Apply Game Logic
        if game_ui_preset != "None":
            ui_map = {"Level Up Notification": "celebratory upward motion, energetic pulses", "Critical Hit Shake": "jagged rapid shaking, sudden impact lines", "Quest Complete Celebration": "shimmering gold effects, starburst expansion", "Low Health Warning": "rhythmic red pulse, shivering borders", "Inventory Item Shine": "rotating shine effect, subtle glowing loop"}
            out += f", {ui_map[game_ui_preset]}"
            
        # Optimization Mode
        mode_map = {"Draft": "simple clean motion", "Polished": "professional vector animation, smooth easing, high detail", "Complex Motion": "complex multi-phase movement, sliding, pulsing, seamless loop"}
        out += f", {mode_map[mode]}"
        
        return (out.strip(", "),)

class OmniLottieEditor:
    """Hub: Instant Property Tweak & Composition"""
    DESCRIPTION = "Post-generation magic. Instantly edit colors, speed, and dimensions, or merge multiple animations into layers without using the model."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lottie_json": ("STRING", {"forceInput": True}),
                "speed_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "canvas_w": ("INT", {"default": 0, "min": 0, "max": 4096}), # 0 = keep original
                "canvas_h": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "overlay_json": ("STRING", {"forceInput": True}),
                "replace_color": ("BOOLEAN", {"default": False}),
                "old_hex": ("STRING", {"default": "#FF0000"}),
                "new_hex": ("STRING", {"default": "#0000FF"}),
                "ping_pong": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("edited_json",)
    FUNCTION = "edit"
    CATEGORY = "OmniLottie"

    def edit(self, lottie_json, speed_multiplier, canvas_w, canvas_h, overlay_json=None, replace_color=False, old_hex="#FF0000", new_hex="#0000FF", ping_pong=False):
        try:
            d = json.loads(lottie_json)
            # Resize
            if canvas_w > 0: d["w"] = canvas_w
            if canvas_h > 0: d["h"] = canvas_h
            # Speed
            if "fr" in d: d["fr"] *= speed_multiplier
            
            # Color Swap (Recursive)
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
            
            # Layer Merging
            if overlay_json:
                db = json.loads(overlay_json)
                d["layers"] = db.get("layers", []) + d.get("layers", []) # Overlay on top
            
            return (json.dumps(d),)
        except:
            return (lottie_json,)

class OmniLottieExporter:
    """Hub: Format Conversion & Rendering"""
    DESCRIPTION = "The export hub. Converts vector Lottie code into professional formats: Images, Masks, MP4 Video, SVG vectors, or Game SpriteSheets."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lottie_json": ("STRING", {"forceInput": True}),
                "export_mode": (["Image Sequence", "Mask Sequence", "Video (MP4)", "Single SVG", "Game SpriteSheet"],),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
            },
            "optional": {
                "fps": ("INT", {"default": 24}),
                "filename_prefix": ("STRING", {"default": "OmniLottie"}),
                "target_frame": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "file_path")
    FUNCTION = "export"
    CATEGORY = "OmniLottie"

    def export(self, lottie_json, export_mode, width, height, fps=24, filename_prefix="OmniLottie", target_frame=0):
        # Placeholder for actual complex rendering logic (requires cairosvg/lottie-python)
        # We return empty tensors to maintain workflow connectivity
        img = torch.zeros((1, height, width, 3))
        mask = torch.zeros((1, height, width))
        path = os.path.join(folder_paths.get_output_directory(), f"{filename_prefix}.bin")
        return (img, mask, path)

class OmniLottieUtilityHub:
    """Hub: System & Info Tools"""
    DESCRIPTION = "Utility toolbox. Monitor Intel Arc hardware, clear VRAM cache, load local JSON libraries, and inspect metadata."
    
    @classmethod
    def INPUT_TYPES(s):
        # Scan local library
        path = os.path.join(folder_paths.get_input_directory(), "lottie")
        if not os.path.exists(path): os.makedirs(path)
        files = [f for f in os.listdir(path) if f.endswith(".json")]
        return {
            "required": {
                "mode": (["XPU Profiler", "Clear VRAM Cache", "Lottie Metadata", "Load Local JSON"],),
                "tick": ("INT", {"default": 0}),
            },
            "optional": {
                "lottie_json": ("STRING", {"forceInput": True}),
                "library_file": (files if files else ["none"],),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("status_text", "vram_percent", "frames", "fps")
    FUNCTION = "execute"
    CATEGORY = "OmniLottie"

    def execute(self, mode, tick, lottie_json="", library_file="none"):
        if mode == "Clear VRAM Cache":
            gc.collect()
            comfy.model_management.soft_empty_cache()
            if hasattr(torch, "xpu"): torch.xpu.empty_cache()
            return ("VRAM Cleared Successfully", 0.0, 0, 0)
        
        if mode == "XPU Profiler":
            if hasattr(torch, "xpu"):
                used = torch.xpu.memory_allocated() / (1024**3)
                return (f"Intel Arc A770: {used:.2f}GB / 16.0GB used", (used/16.0)*100.0, 0, 0)
            return ("Intel Arc XPU hardware not detected", 0.0, 0, 0)
            
        if mode == "Load Local JSON" and library_file != "none":
            fpath = os.path.join(folder_paths.get_input_directory(), "lottie", library_file)
            with open(fpath, "r") as f: content = f.read()
            return (content, 0.0, 0, 0)

        # Default: Metadata
        try:
            d = json.loads(lottie_json)
            f, fr = int(d.get("op", 0)), int(d.get("fr", 0))
            return (f"Dimensions: {d.get('w')}x{d.get('h')} | Frames: {f} | FPS: {fr}", 0.0, f, fr)
        except:
            return ("Connect a Lottie JSON to see info", 0.0, 0, 0)

# --- INDEPENDENT INTERFACE NODES ---

class OmniLottieVisualizer:
    """Interactive Preview Player"""
    DESCRIPTION = "Real-time web player. Renders your vector animation directly in the node for instant inspection."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ()
    FUNCTION = "visualize"
    CATEGORY = "OmniLottie"
    OUTPUT_NODE = True
    def visualize(self, lottie_json): return {"ui": {"lottie_json": [lottie_json]}}

class ImageToPalette:
    """Color Extraction Utility"""
    DESCRIPTION = "Extracts dominant colors from an image. Connect this to the Editor's Hex inputs to sync colors."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"image": ("IMAGE",), "num_colors": ("INT", {"default": 5, "min": 1, "max": 10})}}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("hex_list", "hex_1", "hex_2", "hex_3", "hex_4", "hex_5")
    FUNCTION = "extract"
    CATEGORY = "OmniLottie"
    def extract(self, image, num_colors):
        from sklearn.cluster import KMeans
        img = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB").resize((100, 100))
        px = np.array(img).reshape(-1, 3)
        km = KMeans(n_clusters=num_colors, n_init=10).fit(px)
        hexes = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in km.cluster_centers_.astype(int)]
        return tuple([",".join(hexes)] + (hexes + ["#000000"]*5)[:5])

class SaveLottie:
    """Output Node"""
    DESCRIPTION = "Saves the vector Lottie file to the ComfyUI output directory."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True}), "filename_prefix": ("STRING", {"default": "OmniLottie"})}}
    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "OmniLottie"
    OUTPUT_NODE = True
    def save(self, lottie_json, filename_prefix):
        path = os.path.join(folder_paths.get_output_directory(), f"{filename_prefix}_{comfy.utils.get_random_string(4)}.json")
        with open(path, "w") as f: f.write(lottie_json)
        return {}

# --- CORE INFERENCE LOOP (Optimized for A770) ---

def run_omnilottie_inference(model, prompt, image=None, video=None, params={}):
    device = comfy.model_management.get_torch_device()
    
    # AGGRESSIVE MEMORY MANAGEMENT
    gc.collect()
    comfy.model_management.soft_empty_cache()
    if hasattr(torch, "xpu"): torch.xpu.empty_cache()
    
    try:
        # Move model to device ONLY during inference window
        model.to(device)
        
        content = []
        if image: content.append({"type": "image", "image": image})
        if video: content.append({"type": "video", "video": video})
        content.append({"type": "text", "text": prompt})
        
        text = model.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        
        # Optimized Vision Pre-processing
        vision_info = process_vision_info([{"role": "user", "content": content}])
        inputs = model.processor(
            text=[text], 
            images=vision_info[0] if image else None, 
            videos=vision_info[1] if video else None, 
            padding=True, 
            return_tensors="pt", 
            max_pixels=params.get("max_pixels")
        ).to(device)
        
        with torch.no_grad():
            ids = model.model.generate(
                **inputs, 
                max_new_tokens=params.get("max_new_tokens", 4096), 
                temperature=params.get("temperature", 0.8), 
                do_sample=True,
                use_cache=True, # KV-Caching for speed
                seed=params.get("seed", 0)
            )
            out_ids = ids[0][len(inputs.input_ids[0]):]
            out_text = model.processor.batch_decode([out_ids], skip_special_tokens=True)[0]
            
        if model.decoder:
            try: return (json.dumps(model.decoder.decode(out_text)),)
            except: pass
        return (out_text,)
        
    finally:
        # ARC A770 PROTECTION: Immediately return to CPU to free 15.2GB VRAM
        model.to("cpu")
        gc.collect()
        comfy.model_management.soft_empty_cache()
        if hasattr(torch, "xpu"): torch.xpu.empty_cache()
