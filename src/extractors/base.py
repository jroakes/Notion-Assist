"""
Base extractor class defining the interface for all content extractors.

This module provides an abstract base class that all content extractors must inherit
from and implement. It defines the common interface and shared functionality for
extracting and processing content from various sources.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """
    Data class representing extracted content from any source.

    Attributes:
        content: The main text content extracted
        metadata: Dictionary containing metadata about the extracted content
        raw_data: The raw data from the source before processing
        extracted_at: Timestamp when the content was extracted
        source_url: The URL or identifier of the content source
        errors: List of any errors encountered during extraction
    """

    content: str
    metadata: Dict[str, Any]
    raw_data: Optional[Any] = None
    extracted_at: Optional[datetime] = None
    source_url: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize default values after dataclass initialization."""
        if self.extracted_at is None:
            self.extracted_at = datetime.now(timezone.utc)


class BaseExtractor(ABC):
    """
    Abstract base class for all content extractors.

    This class defines the interface that all extractors must implement and provides
    common functionality such as rate limiting, retry logic, and error handling.

    Attributes:
        name: The name of the extractor
        max_retries: Maximum number of retries for API calls
        retry_delay: Delay in seconds between retries
    """

    def __init__(
        self, name: str, max_retries: int = 3, retry_delay: float = 1.0
    ) -> None:
        """
        Initialize the base extractor.

        Args:
            name: The name of the extractor
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay in seconds between retries
        """
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"Initialized {self.name} extractor")

    @abstractmethod
    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        """
        Extract content from the given source.

        This is the main method that each extractor must implement. It should handle
        the extraction logic specific to each platform/source.

        Args:
            source: The source identifier (URL, ID, path, etc.)
            **kwargs: Additional arguments specific to each extractor

        Returns:
            ExtractedContent object containing the extracted content and metadata

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the extract method")

    @abstractmethod
    def clean_content(self, raw_content: Any) -> str:
        """
        Clean and format the raw content into a standardized string format.

        This method should handle the conversion of platform-specific content
        formats into clean, readable text.

        Args:
            raw_content: The raw content from the source

        Returns:
            Cleaned and formatted content as a string

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the clean_content method")

    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        Validate that the provided source is valid for this extractor.

        This method should check if the source (URL, ID, etc.) is in the correct
        format and can be processed by this extractor.

        Args:
            source: The source identifier to validate

        Returns:
            True if the source is valid, False otherwise

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "Subclasses must implement the validate_source method"
        )

    def preprocess_source(self, source: str) -> str:
        """
        Preprocess the source identifier before extraction.

        This method can be overridden by subclasses to perform any necessary
        preprocessing on the source identifier (e.g., URL normalization).

        Args:
            source: The raw source identifier

        Returns:
            Preprocessed source identifier
        """
        return source.strip()

    def postprocess_content(self, content: str) -> str:
        """
        Postprocess the cleaned content.

        This method can be overridden by subclasses to perform any final
        processing on the cleaned content.

        Args:
            content: The cleaned content

        Returns:
            Postprocessed content
        """
        # Default implementation: strip whitespace and ensure single line breaks
        lines = content.strip().split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if line:  # Only add non-empty lines
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def handle_error(self, error: Exception, context: str) -> None:
        """
        Handle errors in a consistent way across all extractors.

        Args:
            error: The exception that was raised
            context: Additional context about where the error occurred
        """
        error_msg = f"Error in {self.name} extractor ({context}): {str(error)}"
        logger.error(error_msg, exc_info=True)

    def log_extraction_start(self, source: str) -> None:
        """Log the start of an extraction process."""
        logger.info(f"{self.name} extractor: Starting extraction from {source}")

    def log_extraction_complete(self, source: str, content_length: int) -> None:
        """Log the completion of an extraction process."""
        logger.info(
            f"{self.name} extractor: Completed extraction from {source}. "
            f"Content length: {content_length} characters"
        )
