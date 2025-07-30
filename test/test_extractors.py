#!/usr/bin/env python3
"""
Comprehensive test suite for the extractor framework.

This script tests the base extractor functionality and the Notion extractor
implementation without requiring actual API calls.
"""

import sys
import unittest
from unittest.mock import patch
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, "./src")

from src.extractors.base import BaseExtractor, ExtractedContent  # noqa: E402
from src.extractors.notion import NotionExtractor  # noqa: E402


class MockExtractor(BaseExtractor):
    """Mock extractor for testing the base class functionality."""

    def __init__(self):
        super().__init__(name="Mock", max_retries=2, retry_delay=0.5)

    def extract(self, source: str, **kwargs) -> ExtractedContent:
        """Mock extract method."""
        return ExtractedContent(
            content=f"Mock content from {source}",
            metadata={"source": source, "extractor": "mock"},
            source_url=source,
        )

    def clean_content(self, raw_content) -> str:
        """Mock clean_content method."""
        return f"Cleaned: {raw_content}"

    def validate_source(self, source: str) -> bool:
        """Mock validate_source method."""
        return source.startswith("mock://")


class TestExtractedContent(unittest.TestCase):
    """Test cases for ExtractedContent dataclass."""

    def test_initialization(self):
        """Test ExtractedContent initialization."""
        content = ExtractedContent(content="Test content", metadata={"key": "value"})

        self.assertEqual(content.content, "Test content")
        self.assertEqual(content.metadata, {"key": "value"})
        self.assertIsNone(content.raw_data)
        self.assertIsInstance(content.extracted_at, datetime)
        self.assertIsNone(content.source_url)
        self.assertEqual(len(content.errors), 0)

    def test_post_init_defaults(self):
        """Test that post_init sets default values correctly."""
        content = ExtractedContent(content="Test", metadata={})

        self.assertIsNotNone(content.extracted_at)
        self.assertEqual(len(content.errors), 0)


class TestBaseExtractor(unittest.TestCase):
    """Test cases for BaseExtractor base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = MockExtractor()

    def test_initialization(self):
        """Test BaseExtractor initialization."""
        self.assertEqual(self.extractor.name, "Mock")
        self.assertEqual(self.extractor.max_retries, 2)
        self.assertEqual(self.extractor.retry_delay, 0.5)

    def test_extract_method(self):
        """Test extract method works."""
        result = self.extractor.extract("mock://test")

        self.assertIsInstance(result, ExtractedContent)
        self.assertIn("Mock content from mock://test", result.content)
        self.assertEqual(result.metadata["source"], "mock://test")

    def test_preprocess_source(self):
        """Test source preprocessing."""
        result = self.extractor.preprocess_source("  mock://test  ")
        self.assertEqual(result, "mock://test")

    def test_postprocess_content(self):
        """Test content postprocessing."""
        content = "  Line 1  \n\n  Line 2  \n\n  \n  Line 3  "
        result = self.extractor.postprocess_content(content)
        expected = "Line 1\nLine 2\nLine 3"
        self.assertEqual(result, expected)

    def test_validate_source(self):
        """Test source validation."""
        self.assertTrue(self.extractor.validate_source("mock://test"))
        self.assertFalse(self.extractor.validate_source("http://test"))


class TestNotionExtractor(unittest.TestCase):
    """Test cases for NotionExtractor."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the config to avoid needing actual environment variables
        with patch("src.extractors.notion.config") as mock_config:
            mock_config.NOTION_TOKEN = "test_token"
            self.extractor = NotionExtractor()

    def test_initialization(self):
        """Test NotionExtractor initialization."""
        self.assertEqual(self.extractor.name, "Notion")
        self.assertEqual(self.extractor.api_token, "test_token")
        self.assertEqual(self.extractor.api_version, "2022-06-28")
        self.assertIn("Bearer test_token", self.extractor.headers["Authorization"])

    def test_validate_source_url(self):
        """Test source validation for Notion URLs."""
        valid_urls = [
            "https://notion.so/test-page-abc123def456",
            "https://www.notion.so/test-page-abc123def456",
        ]

        for url in valid_urls:
            self.assertTrue(
                self.extractor.validate_source(url), f"Should validate: {url}"
            )

    def test_validate_source_page_id(self):
        """Test source validation for page IDs."""
        valid_ids = [
            "abc123def456789012345678abcdef12",
            "abc12345-def4-5678-9012-345678abcdef",
        ]

        for page_id in valid_ids:
            self.assertTrue(
                self.extractor.validate_source(page_id), f"Should validate: {page_id}"
            )

    def test_validate_source_invalid(self):
        """Test source validation for invalid sources."""
        invalid_sources = ["https://google.com", "not-a-url", "abc123", ""]  # Too short

        for source in invalid_sources:
            self.assertFalse(
                self.extractor.validate_source(source), f"Should not validate: {source}"
            )

    def test_extract_page_id_from_url(self):
        """Test page ID extraction from URLs."""
        test_cases = [
            (
                "https://notion.so/test-page-abc123def456789012345678abcdef12",
                "abc123def456789012345678abcdef12",
            ),
            (
                "https://www.notion.so/My-Page-Title-abc123def456789012345678abcdef12",
                "abc123def456789012345678abcdef12",
            ),
        ]

        for url, expected_id in test_cases:
            result = self.extractor._extract_page_id(url)
            self.assertEqual(result, expected_id, f"Failed for URL: {url}")

    def test_extract_page_id_from_id(self):
        """Test page ID extraction when already an ID."""
        page_id = "abc123def456789012345678abcdef12"
        result = self.extractor._extract_page_id(page_id)
        self.assertEqual(result, page_id)

    def test_extract_rich_text(self):
        """Test rich text extraction."""
        rich_text = [
            {"type": "text", "text": {"content": "Hello "}},
            {"type": "text", "text": {"content": "World"}},
        ]

        result = self.extractor._extract_rich_text(rich_text)
        self.assertEqual(result, "Hello World")

    def test_extract_rich_text_empty(self):
        """Test rich text extraction with empty array."""
        result = self.extractor._extract_rich_text([])
        self.assertEqual(result, "")

    def test_handle_paragraph(self):
        """Test paragraph block handling."""
        block = {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"type": "text", "text": {"content": "This is a paragraph."}}
                ]
            },
        }

        result = self.extractor._handle_paragraph(block)
        self.assertEqual(result, "This is a paragraph.")

    def test_handle_heading_1(self):
        """Test heading 1 block handling."""
        block = {
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"type": "text", "text": {"content": "Main Title"}}]
            },
        }

        result = self.extractor._handle_heading_1(block)
        self.assertEqual(result, "# Main Title")

    def test_handle_bulleted_list(self):
        """Test bulleted list item handling."""
        block = {
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": "List item"}}]
            },
        }

        result = self.extractor._handle_bulleted_list(block)
        self.assertEqual(result, "• List item")

    def test_handle_todo(self):
        """Test todo item handling."""
        # Unchecked todo
        block = {
            "type": "to_do",
            "to_do": {
                "rich_text": [{"type": "text", "text": {"content": "Todo item"}}],
                "checked": False,
            },
        }

        result = self.extractor._handle_todo(block)
        self.assertEqual(result, "[ ] Todo item")

        # Checked todo
        block["to_do"]["checked"] = True
        result = self.extractor._handle_todo(block)
        self.assertEqual(result, "[x] Todo item")

    def test_handle_code(self):
        """Test code block handling."""
        block = {
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": "print('hello')"}}],
                "language": "python",
            },
        }

        result = self.extractor._handle_code(block)
        self.assertEqual(result, "```python\nprint('hello')\n```")

    def test_clean_content(self):
        """Test content cleaning with multiple blocks."""
        blocks = [
            {
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": "Title"}}]
                },
            },
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": "Content"}}]
                },
            },
        ]

        result = self.extractor.clean_content(blocks)
        expected = "# Title\n\nContent"
        self.assertEqual(result, expected)

    @patch("src.extractors.notion.NotionExtractor._fetch_page")
    @patch("src.extractors.notion.NotionExtractor._fetch_all_blocks")
    def test_extract_success(self, mock_fetch_blocks, mock_fetch_page):
        """Test successful extraction."""
        # Mock API responses
        mock_fetch_page.return_value = {
            "id": "test-id",
            "created_time": "2023-01-01T00:00:00.000Z",
            "last_edited_time": "2023-01-01T00:00:00.000Z",
            "properties": {
                "title": {
                    "type": "title",
                    "title": [{"type": "text", "text": {"content": "Test Page"}}],
                }
            },
        }

        mock_fetch_blocks.return_value = [
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": "Test content"}}]
                },
            }
        ]

        result = self.extractor.extract("abc123def456789012345678abcdef12")

        self.assertIsInstance(result, ExtractedContent)
        self.assertEqual(result.content, "Test content")
        self.assertEqual(result.metadata["title"], "Test Page")
        self.assertEqual(len(result.errors), 0)


def run_tests():
    """Run all tests and display results."""
    # Create test suite
    test_classes = [TestExtractedContent, TestBaseExtractor, TestNotionExtractor]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%"
    )

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
