"""Model definitions for SAT.

The local ``sat:tiny`` demo backend is imported eagerly. The Hugging Face
``ModelWithValueHead`` wrapper is loaded lazily so local smoke tests do not
import Transformers unless a HF checkpoint is requested.
"""

from sat.models.tiny_lm import TinyModelWithValueHead, TinyTokenizer, is_tiny_model_path

__all__ = [
    "ModelWithValueHead",
    "TinyModelWithValueHead",
    "TinyTokenizer",
    "is_tiny_model_path",
]


def __getattr__(name: str):
    if name == "ModelWithValueHead":
        from sat.models.causal_lm import ModelWithValueHead

        return ModelWithValueHead
    raise AttributeError(name)
