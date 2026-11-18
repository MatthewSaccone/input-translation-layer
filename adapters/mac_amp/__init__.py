"""
__init__.py — Package Entrypoint for Subfolder Integration
==========================================================
Allows a parent Python project to import this folder directly as a package:
    from <folder_name> import SpecKitTranslator, MacAmpConfig, TranslationResult
"""

from .schema import MacAmpConfig, TranslationResult
from .translator import SpecKitTranslator

__all__ = ["SpecKitTranslator", "MacAmpConfig", "TranslationResult"]
