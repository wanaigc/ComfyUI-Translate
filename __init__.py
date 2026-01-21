from .nodes.gemma import TranslateGemmaNode

# 节点类映射
NODE_CLASS_MAPPINGS = {"Wan_TranslateGemma": TranslateGemmaNode}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {"Wan_TranslateGemma": "Translate (Gemma 4B)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
