"""modernGPT: a modern decoder-only transformer, taken through the full LLM lifecycle."""
from .config import PRESETS, GPTConfig
from .model import ModernGPT

__all__ = ["GPTConfig", "PRESETS", "ModernGPT"]
__version__ = "0.1.0"
