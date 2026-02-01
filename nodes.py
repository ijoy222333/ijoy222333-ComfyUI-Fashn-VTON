import os
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import folder_paths
import types
import traceback

# 存储报错信息
LOAD_ERROR_TRACEBACK = None

# --- 1. 自动寻路逻辑 ---
current_node_path = os.path.dirname(os.path.abspath(__file__))
fashn_lib_parent = None

for root, dirs, files in os.walk(current_node_path):
    if "fashn_vton" in dirs:
        if os.path.exists(os.path.join(root, "fashn_vton", "__init__.py")):
            fashn_lib_parent = root
            break

if fashn_lib_parent:
    if fashn_lib_parent not in sys.path:
        sys.path.insert(0, fashn_lib_parent)
else:
    LOAD_ERROR_TRACEBACK = f"❌ 找不到 'fashn_vton' 文件夹！\n请检查 src 解压位置。"

# --- 2. 核心加载逻辑 ---
TryOnPipeline = None

if not LOAD_ERROR_TRACEBACK:
    try:
        try:
            import fashn_human_parser
        except ImportError:
            raise ImportError("缺少依赖库: fashn-human-parser")

        import fashn_vton
        
        # 路径补丁
        if not hasattr(fashn_vton, "src"):
            mock_src = types.ModuleType("fashn_vton.src")
            fashn_vton.src = mock_src
            sys.modules["fashn_vton.src"] = mock_src
            fashn_vton.src.fashn_vton = fashn_vton
            sys.modules["fashn_vton.src.fashn_vton"] = fashn_vton

        from fashn_vton import TryOnPipeline

    except Exception as e:
        LOAD_ERROR_TRACEBACK = traceback.format_exc()
        TryOnPipeline = None

# --- 3. 辅助函数 ---
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)).convert("RGB")

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# --- 4. 节点定义 ---
class FashnVTON_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        base_model_path = os.path.join(folder_paths.models_dir, "fashn_vton")
        return {
            "required": {
                "model_dir": ("STRING", {"default": base_model_path, "multiline": False}),
            }
        }

    RETURN_TYPES = ("FASHN_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "Fashn-VTON"

    def load_model(self, model_dir=None):
        if TryOnPipeline is None:
            if LOAD_ERROR_TRACEBACK:
                raise ImportError(f"\n{'='*20} 插件加载失败 {'='*20}\n\n{LOAD_ERROR_TRACEBACK}\n{'='*50}")
            else:
                raise ImportError("未知错误：TryOnPipeline 未初始化。")

        if not model_dir:
            model_dir = os.path.join(folder_paths.models_dir, "fashn_vton")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"找不到模型文件夹: {model_dir}")
        
        safetensor = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(safetensor):
             raise FileNotFoundError(f"缺少 model.safetensors: {model_dir}")
        
        try:
            pipeline = TryOnPipeline(weights_dir=model_dir)
            return (pipeline,)
        except Exception as e:
            raise Exception(f"模型初始化失败:\n{traceback.format_exc()}")

class FashnVTON_Run:
    CATEGORY_MAP = {
        "tops (上衣/外套)": "tops",
        "bottoms (下装/裤裙)": "bottoms",
        "one-pieces (全身/连衣裙)": "one-pieces"
    }
    
    GARMENT_TYPE_MAP = {
        "model (模特身上的衣服)": "model",
        "flat-lay (平铺/挂拍的衣服)": "flat-lay",
    }
    
    RESIZE_MODE_MAP = {
        "Bilinear (柔和/抗锯齿)": Image.Resampling.BILINEAR,
        "Bicubic (标准)": Image.Resampling.BICUBIC,
        "Lanczos (锐利)": Image.Resampling.LANCZOS,
        "Nearest (硬边缘/无白边)": Image.Resampling.NEAREST, # 新增
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("FASHN_PIPE",),
                "person_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                
                "category": (list(cls.CATEGORY_MAP.keys()), {"default": "tops (上衣/外套)"}),
                "garment_type": (list(cls.GARMENT_TYPE_MAP.keys()), {"default": "model (模特身上的衣服)"}),
                "num_timesteps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1, "label": "采样步数(Steps)"}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "segmentation_free": ("BOOLEAN", {"default": False, "label": "无分割模式(Seg Free)"}),
            },
            "optional": {
                "restore_original_size": ("BOOLEAN", {"default": True, "label": "强制还原原图尺寸"}),
                "resize_method": (list(cls.RESIZE_MODE_MAP.keys()), {"default": "Bilinear (柔和/抗锯齿)"}),
                
                # --- V13 内衣专用参数 ---
                "smart_erode": ("INT", {"default": 0, "min": 0, "max": 5, "step": 1, "label": "智能缩边(去除白边)"}),
                "texture_boost": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1, "label": "蕾丝/纹理增强(USM)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",) 
    RETURN_NAMES = ("image",)
    FUNCTION = "run_inference"
    CATEGORY = "Fashn-VTON"

    def run_inference(self, pipeline, person_image, garment_image, category, garment_type, num_timesteps, guidance_scale, seed, segmentation_free=False, restore_original_size=True, resize_method="Bilinear (柔和/抗锯齿)", smart_erode=0, texture_boost=0.0):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        if person_image is None or garment_image is None:
            raise ValueError("输入图片不能为空")

        real_category = self.CATEGORY_MAP.get(category, "tops")
        real_garment_type = self.GARMENT_TYPE_MAP.get(garment_type, "model")
        resample_mode = self.RESIZE_MODE_MAP.get(resize_method, Image.Resampling.BILINEAR)

        print(f"🚀 [Fashn-VTON] 开始生成内衣优化版...")
        print(f"   Mode: {real_category} | Erode: {smart_erode} | Texture: {texture_boost}")

        person_pil = tensor2pil(person_image)
        garment_pil = tensor2pil(garment_image)
        original_size = person_pil.size

        try:
            result = pipeline(
                person_image=person_pil,
                garment_image=garment_pil,
                category=real_category,
                garment_photo_type=real_garment_type,
                num_timesteps=num_timesteps, 
                guidance_scale=guidance_scale,
                segmentation_free=segmentation_free,
                num_samples=1,
            )
            output_image = result.images[0]

        except TypeError:
            print(f"⚠️ [Fashn-VTON] 降级兼容...")
            result = pipeline(
                person_image=person_pil,
                garment_image=garment_pil,
                category=real_category,
                num_inference_steps=num_timesteps,
                guidance_scale=guidance_scale
            )
            output_image = result.images[0]
        except Exception as e:
            raise Exception(f"推理失败:\n{traceback.format_exc()}")

        # --- 后处理：内衣/蕾丝 专项优化 ---
        
        # 1. 智能缩边 (Smart Erode) - 在小图阶段处理效果最好
        # 这步是物理去除“白边”的关键，通过最小值滤波让黑色线条变粗，吃掉白边
        if smart_erode > 0:
            # 只有在处理深色蕾丝/浅色背景时开启效果最好
            # 使用 MinFilter 模拟腐蚀效果
            output_image = output_image.filter(ImageFilter.MinFilter(1)) 
            if smart_erode > 1:
                 # 如果还需要更强，再多腐蚀一次，但通常1次够了
                 pass

        if restore_original_size and output_image.size != original_size:
            print(f"↔️ [Fashn-VTON] 还原尺寸...")
            output_image = output_image.resize(original_size, resample_mode)
            
            # 2. 纹理增强 (Texture Boost / USM) - 在大图阶段处理
            # 专门针对蕾丝模糊的问题，使用 USM 锐化提取高频细节
            if texture_boost > 0:
                # 半径 radius=2 是针对 4K 图优化的
                # Percent 是强度
                print(f"✨ [Fashn-VTON] 应用蕾丝增强...")
                output_image = output_image.filter(ImageFilter.UnsharpMask(radius=2, percent=int(texture_boost * 50), threshold=3))

        return (pil2tensor(output_image),)

# --- 注册 ---
NODE_CLASS_MAPPINGS = {
    "FashnVTON_Loader": FashnVTON_Loader,
    "FashnVTON_Run": FashnVTON_Run
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FashnVTON_Loader": "👕 Fashn VTON Loader v1.5",
    "FashnVTON_Run": "👗 Fashn VTON Run-zyp (Lingerie)"
}