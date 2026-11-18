"""
__init__.py — Package Entrypoint for Subfolder Integration
==========================================================
Allows a parent Python project to import this folder directly as a package:
    from adapters.mac_amp import SpecKitTranslator, MacAmpAdapter, MacAmpConfig, TranslationResult
"""

from .schema import BACTERIUM_CODE_MAP, SUPPORTED_BACTERIA, MacAmpConfig, TranslationResult
from .translator import MacAmpAdapter, SpecKitTranslator

__all__ = [
    "SpecKitTranslator",
    "MacAmpAdapter",
    "MacAmpConfig",
    "TranslationResult",
    "BACTERIUM_CODE_MAP",
    "SUPPORTED_BACTERIA",
]
