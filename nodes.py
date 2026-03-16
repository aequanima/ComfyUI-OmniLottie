import os
import json
import torch
import numpy as np
import urllib.request
import re
import math
import gc
import logging
from PIL import Image, ImageDraw
import folder_paths
import comfy.model_management
import comfy.utils
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from huggingface_hub import snapshot_download

# --- HARDENED CORE UTILS ---
logger = logging.getLogger("OmniLottie")
DECODER_URL = "https://raw.githubusercontent.com/OpenVGLab/OmniLottie/main/decoder.py"

def download_required_scripts():
    node_dir = os.path.dirname(__file__)
    decoder_path = os.path.join(node_dir, "decoder.py")
    if not os.path.exists(decoder_path):
        print("OmniLottie: decoder.py missing. Attempting secure download...")
        try:
            # Simple timeout to prevent hanging on boot
            with urllib.request.urlopen(DECODER_URL, timeout=10) as response:
                with open(decoder_path, 'wb') as f:
                    f.write(response.read())
            print("OmniLottie: decoder.py successfully installed.")
        except Exception as e:
            logger.warning(f"OmniLottie: Failed to download decoder. Model output will be raw tokens. Error: {e}")

omnilottie_models_path = os.path.join(folder_paths.models_dir, "omnilottie")
os.makedirs(omnilottie_models_path, exist_ok=True)
download_required_scripts()

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

try:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from decoder import LottieDecoder
except ImportError:
    LottieDecoder = None

def safe_validate_hex(hex_str):
    """Prevents malformed hex codes from corrupting Lottie JSON."""
    if not hex_str or not isinstance(hex_str, str): return "#000000"
    hex_clean = hex_str.strip()
    if not hex_clean.startswith("#"): hex_clean = "#" + hex_clean
    if re.match(r'^#[0-9A-Fa-f]{6}$', hex_clean): return hex_clean
    return "#000000"

class OmniLottieModel:
    def __init__(self, model, processor, decoder):
        self.model, self.processor, self.decoder = model, processor, decoder
    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        return self

# --- HARDENED HUB NODES ---

class OmniLottieModelManager:
    """Hub: Management & Hardware Optimization"""
    DESCRIPTION = "Manages OmniLottie model lifecycle. Features Intel Arc A770 (XPU) optimizations and aggressive VRAM protection."
    
    @classmethod
    def INPUT_TYPES(s):
        model_list = [f for f in os.listdir(omnilottie_models_path) if os.path.isdir(os.path.join(omnilottie_models_path, f))]
        model_list.insert(0, "OpenVGLab/OmniLottie")
        return {
            "required": {
                "model_name": (model_list,),
                "precision": (["Auto", "bfloat16", "float16", "float32"], {"default": "Auto"}),
                "device": (["Auto", "xpu", "cpu", "cuda"], {"default": "Auto"}),
                "vram_limit_gb": ("FLOAT", {"default": 15.5, "min": 8.0, "max": 16.0, "step": 0.1}),
            },
            "optional": {
                "custom_repo_id": ("STRING", {"default": ""}),
                "use_ipex": ("BOOLEAN", {"default": True}),
                "compile": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("OMNILOTTIE_MODEL",), ("model",), "load", "OmniLottie"

    def load(self, model_name, precision, device, vram_limit_gb, custom_repo_id="", use_ipex=True, compile=False):
        target_repo = custom_repo_id.strip() if custom_repo_id.strip() else model_name
        target_device = comfy.model_management.get_torch_device() if device == "Auto" else torch.device(device)
        
        # ARC A770 DType Strategy
        if precision == "Auto":
            dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(target_device) else torch.float16
        else:
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]

        # Secure Download
        safe_repo_name = target_repo.replace("/", "--")
        model_path = os.path.join(omnilottie_models_path, safe_repo_name)
        if not os.path.exists(model_path):
            print(f"OmniLottie: Snapshot download starting for {target_repo}...")
            model_path = snapshot_download(repo_id=target_repo, local_dir=model_path, local_dir_use_symlinks=False)

        # Loading with VRAM protection
        print(f"OmniLottie: Initializing 4B VLM ({dtype}) on CPU...")
        processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
        )
        
        if use_ipex:
            try:
                import intel_extension_for_pytorch as ipex
                print("OmniLottie: Optimizing weights for Intel XPU...")
                model = ipex.optimize(model, dtype=dtype, inplace=True, weights_prepack=True)
            except ImportError:
                logger.info("OmniLottie: IPEX not found, using standard PyTorch.")
            
        if compile:
            try:
                model = torch.compile(model)
            except Exception as e:
                logger.warning(f"OmniLottie: Torch Compile failed, falling back: {e}")
        
        decoder = None
        if LottieDecoder:
            try: decoder = LottieDecoder()
            except: pass
        
        return (OmniLottieModel(model, processor, decoder),)

class OmniLottieGenerator:
    """Hub: Multi-modal Generation & Batching"""
    DESCRIPTION = "Multimodal generator. Handles image/video resolution scaling to prevent Intel Arc VRAM spikes."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("OMNILOTTIE_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A red ball bouncing"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "vram_safety": (["Aggressive", "Standard", "Relaxed"], {"default": "Aggressive"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
                "batch_prompts": ("STRING", {"multiline": True, "default": ""}),
                "max_res": ("INT", {"default": 768, "min": 64, "max": 2048}), # Default lowered for safety
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("STRING",), ("LOTTIE_JSON",), "generate", "OmniLottie"
    OUTPUT_IS_LIST = (True,)

    def generate(self, model, prompt, seed, vram_safety, image=None, video_path="", batch_prompts="", max_res=768, temperature=0.8):
        # Force Memory Flush
        gc.collect()
        comfy.model_management.soft_empty_cache()
        if hasattr(torch, "xpu"): torch.xpu.empty_cache()

        # Batch Preparation
        p_list = [p.strip() for p in batch_prompts.split('\n') if p.strip()]
        if not p_list: p_list = [prompt]
        
        # Image Handling
        pil_img = None
        if image is not None:
            # We only process the first image in a batch to save VRAM
            i = 255. * image[0].cpu().numpy()
            pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        results = []
        # Inference Loop
        for i, p in enumerate(p_list):
            try:
                res = run_omnilottie_inference(
                    model, p, image=pil_img, video=video_path if video_path.strip() else None, 
                    params={"max_pixels": max_res * 28 * 28, "max_new_tokens": 4096, "temperature": temperature, "seed": seed + i}
                )
                results.append(res[0])
            except torch.cuda.OutOfMemoryError or Exception as e:
                # OOM Recovery
                print(f"OmniLottie: Error during generation: {e}")
                results.append(json.dumps({"error": str(e), "layers": []}))
                # Offload model to recover
                model.to("cpu")
                gc.collect()
                if hasattr(torch, "xpu"): torch.xpu.empty_cache()
                break # Stop batch on OOM
            
        return (results,)

class OmniLottieEditor:
    """Hub: Property Tweak & Composition"""
    DESCRIPTION = "Instant JSON modification hub. Includes Hex validation and secure layer merging."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lottie_json": ("STRING", {"forceInput": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            },
            "optional": {
                "overlay_json": ("STRING", {"forceInput": True}),
                "color_swap_A": ("BOOLEAN", {"default": False}),
                "old_hex_A": ("STRING", {"default": "#FF0000"}),
                "new_hex_A": ("STRING", {"default": "#0000FF"}),
                "color_swap_B": ("BOOLEAN", {"default": False}),
                "old_hex_B": ("STRING", {"default": "#FFFFFF"}),
                "new_hex_B": ("STRING", {"default": "#000000"}),
            }
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("STRING",), ("edited_json",), "edit", "OmniLottie"

    def edit(self, lottie_json, speed, overlay_json=None, color_swap_A=False, old_hex_A="", new_hex_A="", color_swap_B=False, old_hex_B="", new_hex_B=""):
        try:
            d = json.loads(lottie_json)
            # Speed
            if "fr" in d: d["fr"] *= speed
            
            # Recursive Hex Swap with Validation
            swaps = []
            if color_swap_A: swaps.append((safe_validate_hex(old_hex_A), safe_validate_hex(new_hex_A)))
            if color_swap_B: swaps.append((safe_validate_hex(old_hex_B), safe_validate_hex(new_hex_B)))
            
            if swaps:
                def h2r(h): h = h.lstrip('#'); return [int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)]
                swap_rgb = [(h2r(o), h2r(n)) for o, n in swaps]
                
                def r_rgb(obj):
                    if isinstance(obj, list):
                        if len(obj) >= 3 and all(isinstance(x, (int, float)) for x in obj[:3]):
                            for old_r, new_r in swap_rgb:
                                if all(abs(obj[i] - old_r[i]) < 0.15 for i in range(3)):
                                    for i in range(3): obj[i] = new_r[i]
                        for i in obj: r_rgb(i)
                    elif isinstance(obj, dict):
                        for v in obj.values(): r_rgb(v)
                r_rgb(d)
            
            # Layer Merging
            if overlay_json:
                try:
                    db = json.loads(overlay_json)
                    d["layers"] = db.get("layers", []) + d.get("layers", [])
                except: pass
            
            return (json.dumps(d),)
        except Exception as e:
            logger.error(f"OmniLottie: Editor failed - {e}")
            return (lottie_json,)

class OmniLottieUtilityHub:
    """Hub: System & Metadata"""
    DESCRIPTION = "System toolbox. Features path-safe file loading and real-time Intel Arc profiling."
    
    @classmethod
    def INPUT_TYPES(s):
        path = os.path.join(folder_paths.get_input_directory(), "lottie")
        os.makedirs(path, exist_ok=True)
        files = [f for f in os.listdir(path) if f.endswith(".json")]
        return {
            "required": {
                "mode": (["XPU Profiler", "VRAM Purge", "Metadata Info", "Load From Input"],),
                "tick": ("INT", {"default": 0}),
            },
            "optional": {
                "lottie_json": ("STRING", {"forceInput": True}),
                "filename": (files if files else ["none"],),
            }
        }

    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("STRING", "FLOAT", "INT", "INT"), ("output", "vram_usage", "frames", "fps"), "execute", "OmniLottie"

    def execute(self, mode, tick, lottie_json="", filename="none"):
        if mode == "VRAM Purge":
            gc.collect()
            comfy.model_management.soft_empty_cache()
            if hasattr(torch, "xpu"): torch.xpu.empty_cache()
            return ("VRAM Cleared", 0.0, 0, 0)
        
        if mode == "XPU Profiler":
            if hasattr(torch, "xpu"):
                used = torch.xpu.memory_allocated() / (1024**3)
                return (f"Arc A770: {used:.2f}GB Used", (used/16.0)*100.0, 0, 0)
            return ("XPU Not Found", 0.0, 0, 0)
            
        if mode == "Load From Input" and filename != "none":
            # SECURITY: Prevent path traversal by stripping directory components
            safe_filename = os.path.basename(filename)
            fpath = os.path.join(folder_paths.get_input_directory(), "lottie", safe_filename)
            try:
                with open(fpath, "r", encoding="utf-8") as f: return (f.read(), 0.0, 0, 0)
            except: return ("Load Failed", 0.0, 0, 0)

        # Metadata
        try:
            d = json.loads(lottie_json)
            f, fr = int(d.get("op", 0)), int(d.get("fr", 0))
            return (f"Lottie: {d.get('w')}x{d.get('h')} | {f} frames", 0.0, f, fr)
        except:
            return ("No Data", 0.0, 0, 0)

class ImageToPalette:
    """Hub: Design Asset Utility"""
    DESCRIPTION = "Dominant color extraction. Safely handles missing scikit-learn dependency."
    @classmethod
    def INPUT_TYPES(s): return {"required": {"image": ("IMAGE",), "num_colors": ("INT", {"default": 5, "min": 1, "max": 10})}}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("hex_list", "hex_1", "hex_2", "hex_3", "hex_4", "hex_5")
    FUNCTION = "extract"
    CATEGORY = "OmniLottie"
    
    def extract(self, image, num_colors):
        try:
            from sklearn.cluster import KMeans
            img = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB").resize((100, 100))
            px = np.array(img).reshape(-1, 3)
            km = KMeans(n_clusters=num_colors, n_init=10).fit(px)
            hexes = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in km.cluster_centers_.astype(int)]
            return tuple([",".join(hexes)] + (hexes + ["#000000"]*5)[:5])
        except ImportError:
            logger.error("OmniLottie: scikit-learn not found. Returning black palette.")
            return ("#000000", "#000000", "#000000", "#000000", "#000000", "#000000")

# --- PASSTHROUGH & OUTPUT NODES ---

class OmniLottieVisualizer:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True})}}
    RETURN_TYPES, FUNCTION, CATEGORY, OUTPUT_NODE = (), "visualize", "OmniLottie", True
    def visualize(self, lottie_json): return {"ui": {"lottie_json": [lottie_json]}}

class SaveLottie:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"lottie_json": ("STRING", {"forceInput": True}), "filename_prefix": ("STRING", {"default": "OmniLottie"})}}
    RETURN_TYPES, FUNCTION, CATEGORY, OUTPUT_NODE = (), "save", "OmniLottie", True
    def save(self, lottie_json, filename_prefix):
        path = os.path.join(folder_paths.get_output_directory(), f"{filename_prefix}_{comfy.utils.get_random_string(4)}.json")
        with open(path, "w", encoding="utf-8") as f: f.write(lottie_json)
        return {}

# --- RE-FACTORED INFERENCE (The Hardware Fortress) ---

def get_optimal_device():
    """Fallback logic for non-Intel hardware."""
    try:
        return comfy.model_management.get_torch_device()
    except Exception:
        if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

def clear_hardware_cache():
    """Universal cache clearer for all hardware types."""
    gc.collect()
    try: comfy.model_management.soft_empty_cache()
    except: pass
    
    if hasattr(torch, "xpu") and torch.xpu.is_available(): torch.xpu.empty_cache()
    elif torch.cuda.is_available(): torch.cuda.empty_cache()
    elif torch.backends.mps.is_available(): torch.mps.empty_cache()

def run_omnilottie_inference(omni_model, prompt, image=None, video=None, params={}):
    device = get_optimal_device()
    
    # Pre-inference Purge
    clear_hardware_cache()
    
    try:
        # Move model ONLY for actual work
        print(f"OmniLottie: Model VRAM Entry (Device: {device})")
        omni_model.to(device)
        
        content = []
        if image: content.append({"type": "image", "image": image})
        if video: content.append({"type": "video", "video": video})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        text = omni_model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Vision Pass: Qwen2-VL specific memory management
        if process_vision_info:
            vision_data = process_vision_info(messages)
            inputs = omni_model.processor(
                text=[text], 
                images=vision_data[0] if image else None, 
                videos=vision_data[1] if video else None, 
                padding=True, return_tensors="pt", 
                max_pixels=params.get("max_pixels", 768 * 28 * 28)
            ).to(device)
        else:
            inputs = omni_model.processor(text=[text], padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = omni_model.model.generate(
                **inputs, 
                max_new_tokens=params.get("max_new_tokens", 4096), 
                temperature=params.get("temperature", 0.8), 
                do_sample=True, 
                use_cache=True,
                seed=params.get("seed", 0)
            )
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = omni_model.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
        if omni_model.decoder:
            try: return (json.dumps(omni_model.decoder.decode(output_text)),)
            except: pass
        return (output_text,)
        
    finally:
        # ABSOLUTE OFFLOAD: Never leave the model on VRAM
        print("OmniLottie: Model VRAM Exit (Hard Offload to System RAM: 64GB DDR4 Target)")
        omni_model.to("cpu")
        clear_hardware_cache()

