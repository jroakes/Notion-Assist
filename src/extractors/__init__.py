"""
Extractors package for content extraction from various sources.

This package provides a framework for extracting content from different platforms
such as Notion, Google Docs, etc. All extractors follow a common interface defined
in the base extractor class.
"""

from .base import BaseExtractor
from .notion import NotionExtractor

__all__ = ["BaseExtractor", "NotionExtractor"]
