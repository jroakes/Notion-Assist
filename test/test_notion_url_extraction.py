#!/usr/bin/env python3
"""
Test script for Notion URL extraction.

This script accepts a Notion URL as a command-line argument and uses the
Notion extractor to pull and display the content.

Usage:
    python test_notion_url_extraction.py <notion_url>

Example:
    python test_notion_url_extraction.py "https://notion.so/mypage-abc123def456"
"""

import argparse
import json
import logging
import sys

# Add the src directory to the Python path
sys.path.insert(0, "./src")

from src.extractors.notion import NotionExtractor  # noqa: E402
from src.extractors.base import ExtractedContent  # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        verbose: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level, format=format_str, handlers=[logging.StreamHandler(sys.stdout)]
    )


def print_extracted_content(content: ExtractedContent) -> None:
    """
    Print the extracted content in a formatted way.

    Args:
        content: ExtractedContent object to display
    """
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80)

    # Print metadata
    print("\nMETADATA:")
    print("-" * 40)
    if content.metadata:
        for key, value in content.metadata.items():
            if key != "properties":  # Handle properties separately
                print(f"{key}: {value}")

        # Print properties if available
        if "properties" in content.metadata and content.metadata["properties"]:
            print("\nPROPERTIES:")
            for prop_name, prop_value in content.metadata["properties"].items():
                print(f"  {prop_name}: {prop_value}")
    else:
        print("No metadata available")

    # Print content
    print("\n" + "-" * 40)
    print("CONTENT:")
    print("-" * 40)
    if content.content:
        print(content.content)
    else:
        print("No content extracted")

    # Print errors if any
    if content.errors:
        print("\n" + "-" * 40)
        print("ERRORS:")
        print("-" * 40)
        for error in content.errors:
            print(f"- {error}")

    print("\n" + "=" * 80)

    # Print summary statistics
    print("\nSUMMARY:")
    print(f"- Content length: {len(content.content)} characters")
    print(f"- Extracted at: {content.extracted_at}")
    print(f"- Source URL: {content.source_url}")
    print(f"- Errors encountered: {len(content.errors)}")


def save_raw_data(content: ExtractedContent, output_file: str) -> None:
    """
    Save raw data to a JSON file for debugging.

    Args:
        content: ExtractedContent object containing raw data
        output_file: Path to save the JSON file
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(content.raw_data, f, indent=2, ensure_ascii=False)
        print(f"\nRaw data saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving raw data: {e}")


def main() -> int:
    """
    Main function to run the Notion URL extraction test.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Test Notion URL extraction functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_notion_url_extraction.py "https://notion.so/mypage-abc123def456"
  python test_notion_url_extraction.py "abc123def456" --verbose
  python test_notion_url_extraction.py "https://notion.so/mypage-abc123def456" --save-raw data.json
        """,
    )

    parser.add_argument(
        "notion_url", help="Notion page URL or page ID to extract content from"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "-s",
        "--save-raw",
        metavar="FILE",
        help="Save raw API response data to a JSON file",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Create Notion extractor
        logger.info("Initializing Notion extractor...")
        extractor = NotionExtractor()

        # Extract content
        logger.info(f"Extracting content from: {args.notion_url}")
        extracted_content = extractor.extract(args.notion_url)

        # Print results
        print_extracted_content(extracted_content)

        # Save raw data if requested
        if args.save_raw:
            save_raw_data(extracted_content, args.save_raw)

        # Return success if content was extracted
        if extracted_content.content:
            return 0
        else:
            logger.warning("No content was extracted")
            return 1

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nError: {e}")
        print("Please ensure your NOTION_TOKEN is set in the .env file")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
