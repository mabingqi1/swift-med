from typing import List
from dataclasses import dataclass, field
from easydict import EasyDict


@dataclass
class InferArguments(EasyDict):
    model_id_or_path: str = None
    adapters: List[str] = None
    max_batch_size: int = 32
    attn_impl: str = "flash_attn"
    video_name: str = "zst_path"
    image_name: str = None
    pred_name: str = "pred_llm"
    temperature: float = 0.05
    max_tokens: int = 2048
    system: str = "You are a professional medical assistant specialized in analyzing medical imaging and generating accurate clinical reports."
    prompt: str = "<video>Please extract Findings and Impression from medical video."