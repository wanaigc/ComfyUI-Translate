import os
import torch
import folder_paths
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# --------------------------------------------------------------------------------
# 全局缓存变量
# --------------------------------------------------------------------------------
GLOBAL_MODEL_CACHE = {"model": None, "processor": None, "current_model_path": None}

# --------------------------------------------------------------------------------
# 常用语言代码列表
# 格式：代码 (名称) - 方便用户查看，程序会自动截取空格前的代码
# --------------------------------------------------------------------------------
LANGUAGE_OPTIONS = [
    "en (英文 - English)",  # 放在前面方便查找
    "zh (中文 - Chinese)",
    "ja (日文 - Japanese)",
    "ko (韩文 - Korean)",
    "fr (法文 - French)",
    "de (德文 - German)",
    "es (西班牙文 - Spanish)",
    "ru (俄文 - Russian)",
    "it (意大利文 - Italian)",
    "pt (葡萄牙文 - Portuguese)",
    "ar (阿拉伯文 - Arabic)",
    "hi (印地文 - Hindi)",
    "th (泰文 - Thai)",
    "vi (越南文 - Vietnamese)",
    "id (印尼文 - Indonesian)",
    "cs (捷克文 - Czech)",
    "nl (荷兰文 - Dutch)",
    "pl (波兰文 - Polish)",
    "tr (土耳其文 - Turkish)",
    "uk (乌克兰文 - Ukrainian)",
    "de-DE (德文-德国)",
    "en-US (英文-美国)",
    "en-GB (英文-英国)",
    "zh-CN (简体中文)",
    "zh-TW (繁体中文)",
]


class TranslateGemmaNode:
    """
    Google TranslateGemma ComfyUI 节点
    支持文本翻译与图像文本提取翻译
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # 注册自定义的模型路径
        model_dir = os.path.join(folder_paths.models_dir, "Translate")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        model_list = [
            f
            for f in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, f))
        ]
        if not model_list:
            model_list = ["请将模型放入 models/Translate 目录"]

        return {
            "required": {
                "model_name": (model_list,),
                # --- 源语言设置 (默认 en) ---
                "source_lang_select": (
                    LANGUAGE_OPTIONS,
                    {"default": "en (英文 - English)"},
                ),
                "source_manual_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "手动输入代码 (如 sw)，将覆盖上方选项",
                    },
                ),
                # --- 目标语言设置 (默认 zh) ---
                "target_lang_select": (
                    LANGUAGE_OPTIONS,
                    {"default": "zh (中文 - Chinese)"},
                ),
                "target_manual_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "手动输入代码，将覆盖上方选项",
                    },
                ),
                "text_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "在此输入要翻译的文本（如果未连接图像）",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "process"
    CATEGORY = "Translate/Local"

    def load_model(self, model_name):
        """加载模型的逻辑，包含缓存处理"""
        global GLOBAL_MODEL_CACHE

        model_path = os.path.join(folder_paths.models_dir, "Translate", model_name)

        if (
            GLOBAL_MODEL_CACHE["model"] is not None
            and GLOBAL_MODEL_CACHE["current_model_path"] == model_path
        ):
            return GLOBAL_MODEL_CACHE["model"], GLOBAL_MODEL_CACHE["processor"]

        print(f"Loading TranslateGemma model from: {model_path} ...")

        try:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            device_map = "auto"

            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, device_map=device_map, torch_dtype=dtype
            )

            GLOBAL_MODEL_CACHE["model"] = model
            GLOBAL_MODEL_CACHE["processor"] = processor
            GLOBAL_MODEL_CACHE["current_model_path"] = model_path

            return model, processor

        except Exception as e:
            raise RuntimeError(
                f"加载模型失败: {e}\n请确保已安装 transformers>=4.48.0 和 accelerate 库。"
            )

    def process(
        self,
        model_name,
        source_lang_select,
        source_manual_override,
        target_lang_select,
        target_manual_override,
        text_input,
        image=None,
    ):
        """核心推理函数"""
        # 1. 路径校验
        if model_name.startswith("请将模型"):
            raise ValueError("未找到模型，请检查 models/Translate 目录。")

        # ---------------------------------------------------------
        # 语言代码清洗逻辑
        # ---------------------------------------------------------
        def clean_lang_code(selection_str, manual_str):
            # 1. 优先使用手动输入 (Trim 去除空格)
            if manual_str and manual_str.strip():
                return manual_str.strip()

            # 2. 否则使用下拉菜单，并提取第一个空格前的代码
            # 例如: "zh (中文 - Chinese)" -> "zh"
            if selection_str:
                return selection_str.split(" ")[0]

            return "en"  # 默认兜底

        final_source = clean_lang_code(source_lang_select, source_manual_override)
        final_target = clean_lang_code(target_lang_select, target_manual_override)

        # 2. 加载模型
        model, processor = self.load_model(model_name)

        # 3. 构建 Prompt
        messages = []
        pil_image = None

        if image is not None:
            # ---- 图像模式 ----
            image_tensor = image[0]
            i = 255.0 * image_tensor.cpu().numpy()
            pil_image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            content_entry = {
                "type": "image",
                "source_lang_code": final_source,
                "target_lang_code": final_target,
                "url": "https://placeholder/image.jpg",
            }
        else:
            # ---- 文本模式 ----
            if not text_input or text_input.strip() == "":
                raise ValueError("未提供图像且文本输入为空。")

            content_entry = {
                "type": "text",
                "source_lang_code": final_source,
                "target_lang_code": final_target,
                "text": text_input,
            }

        messages = [{"role": "user", "content": [content_entry]}]

        # 4. 推理
        try:
            prompt_str = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            process_kwargs = {"text": prompt_str, "return_tensors": "pt"}
            if pil_image:
                process_kwargs["images"] = pil_image

            inputs = processor(**process_kwargs).to(model.device)
            input_len = inputs.input_ids.shape[1]

            with torch.inference_mode():
                generation = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False
                )

            generated_ids = generation[0][input_len:]
            decoded_text = processor.decode(generated_ids, skip_special_tokens=True)

            return (decoded_text,)

        except Exception as e:
            print(f"TranslateGemma 推理错误: {str(e)}")
            raise e
