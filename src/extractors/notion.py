"""
Notion-specific extractor implementation.

This module provides functionality to extract content from Notion pages using
the Notion API. It handles all Notion block types and converts them to clean,
readable text format.
"""

import re
import time
import logging
from functools import wraps
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
from urllib3.util.retry import Retry

from ..settings import config
from .base import BaseExtractor, ExtractedContent


logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter decorator to prevent API abuse.

    Implements a sliding window rate limiter to ensure API calls
    don't exceed the specified limits.
    """

    def __init__(self, max_calls: int = 3, time_window: int = 1) -> None:
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
        self.lock = Lock()

    def __call__(self, func):
        """Rate limiter decorator."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                # Remove calls outside the time window
                self.calls = [
                    call for call in self.calls if now - call < self.time_window
                ]

                # If we've hit the limit, wait
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.time_window - (now - self.calls[0])
                    if sleep_time > 0:
                        logger.debug(
                            f"Rate limit reached, sleeping for {sleep_time:.2f} seconds"
                        )
                        time.sleep(sleep_time)

                # Record this call
                self.calls.append(now)

            return func(*args, **kwargs)

        return wrapper


def mask_sensitive_data(data: str, sensitive_words: List[str] = None) -> str:
    """
    Mask sensitive information in log messages.

    Args:
        data: The string to mask
        sensitive_words: List of sensitive words to mask

    Returns:
        String with sensitive data masked
    """
    if sensitive_words is None:
        sensitive_words = ["Bearer", "token", "secret", "key", "password"]

    masked_data = data
    for word in sensitive_words:
        if word in masked_data:
            # Replace the sensitive data with asterisks
            masked_data = re.sub(
                rf"{word}\s+[^\s]+", f"{word} ***", masked_data, flags=re.IGNORECASE
            )

    return masked_data


class NotionExtractor(BaseExtractor):
    """
    Extractor for Notion pages.

    This extractor uses the Notion API to fetch page content and converts
    various Notion block types into clean, readable text.

    Attributes:
        api_token: Notion integration token
        api_version: Notion API version
        base_url: Base URL for Notion API
        session: Requests session with retry configuration
    """

    def __init__(self, api_token: Optional[str] = None) -> None:
        """
        Initialize the Notion extractor.

        Args:
            api_token: Notion integration token. If not provided, uses NOTION_TOKEN from config
        """
        super().__init__(name="Notion", max_retries=3, retry_delay=1.0)

        self.api_token = api_token or config.NOTION_TOKEN
        if not self.api_token:
            raise ValueError(
                "Notion API token is required but not found in configuration"
            )

        self.api_version = "2022-06-28"
        self.base_url = "https://api.notion.com/v1"

        # Set up session with retry configuration
        self.session = self._create_session()

        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Notion-Version": self.api_version,
        }

        # Set up rate limiter (3 requests per second to be respectful to Notion API)
        self.rate_limiter = RateLimiter(max_calls=3, time_window=1)

    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry configuration.

        Returns:
            Configured requests session
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=[
                "HEAD",
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "OPTIONS",
                "TRACE",
            ],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def extract(self, source: str, **kwargs: Any) -> ExtractedContent:
        """
        Extract content from a Notion page.

        Args:
            source: Notion page URL or page ID
            **kwargs: Additional arguments (not used currently)

        Returns:
            ExtractedContent object containing the extracted content
        """
        # Log any additional kwargs for debugging
        if kwargs:
            logger.debug(f"Additional kwargs provided: {kwargs}")

        self.log_extraction_start(source)

        # Validate and preprocess source
        if not self.validate_source(source):
            raise ValueError(f"Invalid Notion source: {source}")

        processed_source = self.preprocess_source(source)
        page_id = self._extract_page_id(processed_source)

        errors = []
        raw_data = {}

        try:
            # Fetch page metadata
            page_data = self._fetch_page(page_id)
            raw_data["page"] = page_data

            # Fetch page content (blocks)
            blocks = self._fetch_all_blocks(page_id)
            raw_data["blocks"] = blocks

            # Extract metadata
            metadata = self._extract_metadata(page_data)

            # Clean and format content
            content = self.clean_content(blocks)
            content = self.postprocess_content(content)

            self.log_extraction_complete(source, len(content))

            return ExtractedContent(
                content=content,
                metadata=metadata,
                raw_data=raw_data,
                source_url=source,
                errors=errors,
            )

        except HTTPError as e:
            if e.response and e.response.status_code == 401:
                error_msg = "Invalid Notion API token - please check your NOTION_TOKEN"
            elif e.response and e.response.status_code == 403:
                error_msg = f"Access denied to Notion page: {source}. Please ensure the integration has access."
            elif e.response and e.response.status_code == 404:
                error_msg = f"Notion page not found: {source}"
            else:
                error_msg = f"HTTP error: {e}"

            self.handle_error(e, f"extracting from {source}")
            errors.append(error_msg)

        except (ConnectionError, Timeout) as e:
            error_msg = f"Network error while accessing Notion API: {e}"
            self.handle_error(e, f"extracting from {source}")
            errors.append(error_msg)

        except RequestException as e:
            error_msg = f"Request failed: {e}"
            self.handle_error(e, f"extracting from {source}")
            errors.append(error_msg)

        except ValueError as e:
            error_msg = f"Invalid input: {e}"
            self.handle_error(e, f"extracting from {source}")
            errors.append(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            self.handle_error(e, f"extracting from {source}")
            errors.append(error_msg)

            return ExtractedContent(
                content="",
                metadata={},
                raw_data=raw_data,
                source_url=source,
                errors=errors,
            )

    def clean_content(self, raw_content: Any) -> str:
        """
        Clean and format Notion blocks into readable text.

        Args:
            raw_content: List of Notion blocks

        Returns:
            Cleaned and formatted content as a string
        """
        if not isinstance(raw_content, list):
            return ""

        content_parts = []

        for block in raw_content:
            block_text = self._process_block(block)
            if block_text:
                content_parts.append(block_text)

        return "\n\n".join(content_parts)

    def validate_source(self, source: str) -> bool:
        """
        Validate that the source is a valid Notion URL or page ID.

        Args:
            source: The source to validate

        Returns:
            True if valid, False otherwise
        """
        # Check if it's a valid UUID (page ID)
        uuid_pattern = (
            r"^[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12}$"
        )
        if re.match(uuid_pattern, source.replace("-", ""), re.IGNORECASE):
            return True

        # Check if it's a valid Notion URL
        if source.startswith(("https://notion.so/", "https://www.notion.so/")):
            return True

        return False

    def _extract_page_id(self, source: str) -> str:
        """
        Extract page ID from a Notion URL or return the ID if already provided.

        Args:
            source: Notion URL or page ID

        Returns:
            Page ID
        """
        # If it's already a page ID, return it
        uuid_pattern = (
            r"^[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{12}$"
        )
        if re.match(uuid_pattern, source.replace("-", ""), re.IGNORECASE):
            return source

        # Extract from URL
        parsed = urlparse(source)
        path_parts = parsed.path.strip("/").split("/")

        if not path_parts:
            raise ValueError(f"Could not extract page ID from URL: {source}")

        # The page ID is usually the last part of the URL or after a '-'
        last_part = path_parts[-1]

        # Handle URLs like /Page-Title-abc123def456
        if "-" in last_part:
            # Get the part after the last hyphen
            potential_id = last_part.split("-")[-1]
            if len(potential_id) >= 32:  # Notion IDs are 32 characters without hyphens
                return potential_id

        # Handle URLs with just the ID
        if len(last_part.replace("-", "")) == 32:
            return last_part

        raise ValueError(f"Could not extract valid page ID from URL: {source}")

    @RateLimiter(max_calls=3, time_window=1)
    def _fetch_page(self, page_id: str) -> Dict[str, Any]:
        """
        Fetch page metadata from Notion API.

        Args:
            page_id: The Notion page ID

        Returns:
            Page data from the API
        """
        url = f"{self.base_url}/pages/{page_id}"

        response = self.session.get(url, headers=self.headers)

        if response.status_code == 429:  # Rate limited
            retry_after = int(response.headers.get("Retry-After", self.retry_delay))
            logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            response = self.session.get(url, headers=self.headers)

        response.raise_for_status()
        return response.json()

    @RateLimiter(max_calls=3, time_window=1)
    def _fetch_all_blocks(self, page_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all blocks from a Notion page, handling pagination.

        Args:
            page_id: The Notion page ID

        Returns:
            List of all blocks in the page
        """
        blocks = []
        has_more = True
        start_cursor = None

        while has_more:
            url = f"{self.base_url}/blocks/{page_id}/children"
            params = {"page_size": 100}

            if start_cursor:
                params["start_cursor"] = start_cursor

            response = self.session.get(url, headers=self.headers, params=params)

            if response.status_code == 429:  # Rate limited
                retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                response = self.session.get(url, headers=self.headers, params=params)

            response.raise_for_status()
            data = response.json()

            blocks.extend(data.get("results", []))
            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

        # Recursively fetch children for blocks that have them
        all_blocks = []
        for block in blocks:
            all_blocks.append(block)
            if block.get("has_children", False):
                children = self._fetch_all_blocks(block["id"])
                all_blocks.extend(children)

        return all_blocks

    def _extract_metadata(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from page data.

        Args:
            page_data: Raw page data from Notion API

        Returns:
            Dictionary of metadata
        """
        metadata = {
            "id": page_data.get("id"),
            "created_time": page_data.get("created_time"),
            "last_edited_time": page_data.get("last_edited_time"),
            "archived": page_data.get("archived", False),
            "url": page_data.get("url"),
            "properties": {},
        }

        # Extract properties
        properties = page_data.get("properties", {})
        for prop_name, prop_data in properties.items():
            prop_type = prop_data.get("type")

            if prop_type == "title":
                title_content = prop_data.get("title", [])
                metadata["properties"][prop_name] = self._extract_rich_text(
                    title_content
                )
            elif prop_type == "rich_text":
                text_content = prop_data.get("rich_text", [])
                metadata["properties"][prop_name] = self._extract_rich_text(
                    text_content
                )
            elif prop_type in ["number", "checkbox", "email", "phone_number", "url"]:
                metadata["properties"][prop_name] = prop_data.get(prop_type)
            elif prop_type == "select":
                select_data = prop_data.get("select", {})
                metadata["properties"][prop_name] = (
                    select_data.get("name") if select_data else None
                )
            elif prop_type == "multi_select":
                multi_select_data = prop_data.get("multi_select", [])
                metadata["properties"][prop_name] = [
                    item.get("name") for item in multi_select_data
                ]
            elif prop_type == "date":
                date_data = prop_data.get("date", {})
                if date_data:
                    metadata["properties"][prop_name] = {
                        "start": date_data.get("start"),
                        "end": date_data.get("end"),
                    }
            elif prop_type == "people":
                people_data = prop_data.get("people", [])
                metadata["properties"][prop_name] = [
                    person.get("name") for person in people_data
                ]

        # Set title as main metadata field if available
        for prop_name, prop_data in properties.items():
            if prop_data.get("type") == "title":
                metadata["title"] = metadata["properties"][prop_name]
                break

        return metadata

    def _process_block(self, block: Dict[str, Any]) -> str:
        """
        Process a single Notion block and convert it to text.

        Args:
            block: Notion block data

        Returns:
            Text representation of the block
        """
        block_type = block.get("type")

        if not block_type:
            return ""

        # Handle different block types
        handlers = {
            "paragraph": self._handle_paragraph,
            "heading_1": self._handle_heading_1,
            "heading_2": self._handle_heading_2,
            "heading_3": self._handle_heading_3,
            "bulleted_list_item": self._handle_bulleted_list,
            "numbered_list_item": self._handle_numbered_list,
            "to_do": self._handle_todo,
            "toggle": self._handle_toggle,
            "code": self._handle_code,
            "quote": self._handle_quote,
            "callout": self._handle_callout,
            "divider": self._handle_divider,
            "table": self._handle_table,
            "table_row": self._handle_table_row,
            "column": self._handle_column,
            "column_list": self._handle_column_list,
            "link_to_page": self._handle_link_to_page,
            "image": self._handle_image,
            "video": self._handle_video,
            "file": self._handle_file,
            "pdf": self._handle_pdf,
            "bookmark": self._handle_bookmark,
            "embed": self._handle_embed,
        }

        handler = handlers.get(block_type)
        if handler:
            return handler(block)
        else:
            logger.warning(f"Unhandled block type: {block_type}")
            return ""

    def _extract_rich_text(self, rich_text_array: List[Dict[str, Any]]) -> str:
        """
        Extract plain text from Notion rich text array.

        Args:
            rich_text_array: Array of rich text elements

        Returns:
            Plain text string
        """
        if not rich_text_array:
            return ""

        text_parts = []
        for text_obj in rich_text_array:
            if text_obj.get("type") == "text":
                text_parts.append(text_obj.get("text", {}).get("content", ""))
            elif text_obj.get("type") == "mention":
                mention = text_obj.get("mention", {})
                if mention.get("type") == "page":
                    text_parts.append(
                        f"[Page: {mention.get('page', {}).get('id', '')}]"
                    )
                elif mention.get("type") == "user":
                    text_parts.append(f"@{mention.get('user', {}).get('name', 'User')}")
            elif text_obj.get("type") == "equation":
                text_parts.append(
                    f"${text_obj.get('equation', {}).get('expression', '')}$"
                )

        return "".join(text_parts)

    # Block handlers
    def _handle_paragraph(self, block: Dict[str, Any]) -> str:
        """Handle paragraph blocks."""
        text_content = block.get("paragraph", {}).get("rich_text", [])
        return self._extract_rich_text(text_content)

    def _handle_heading_1(self, block: Dict[str, Any]) -> str:
        """Handle heading 1 blocks."""
        text_content = block.get("heading_1", {}).get("rich_text", [])
        text = self._extract_rich_text(text_content)
        return f"# {text}" if text else ""

    def _handle_heading_2(self, block: Dict[str, Any]) -> str:
        """Handle heading 2 blocks."""
        text_content = block.get("heading_2", {}).get("rich_text", [])
        text = self._extract_rich_text(text_content)
        return f"## {text}" if text else ""

    def _handle_heading_3(self, block: Dict[str, Any]) -> str:
        """Handle heading 3 blocks."""
        text_content = block.get("heading_3", {}).get("rich_text", [])
        text = self._extract_rich_text(text_content)
        return f"### {text}" if text else ""

    def _handle_bulleted_list(self, block: Dict[str, Any]) -> str:
        """Handle bulleted list items."""
        text_content = block.get("bulleted_list_item", {}).get("rich_text", [])
        text = self._extract_rich_text(text_content)
        return f"• {text}" if text else ""

    def _handle_numbered_list(self, block: Dict[str, Any]) -> str:
        """Handle numbered list items."""
        text_content = block.get("numbered_list_item", {}).get("rich_text", [])
        text = self._extract_rich_text(text_content)
        return f"1. {text}" if text else ""

    def _handle_todo(self, block: Dict[str, Any]) -> str:
        """Handle to-do list items."""
        todo_data = block.get("to_do", {})
        text_content = todo_data.get("rich_text", [])
        text = self._extract_rich_text(text_content)
        checked = todo_data.get("checked", False)
        checkbox = "[x]" if checked else "[ ]"
        return f"{checkbox} {text}" if text else ""

    def _handle_toggle(self, block: Dict[str, Any]) -> str:
        """Handle toggle blocks."""
        text_content = block.get("toggle", {}).get("rich_text", [])
        text = self._extract_rich_text(text_content)
        return f"▸ {text}" if text else ""

    def _handle_code(self, block: Dict[str, Any]) -> str:
        """Handle code blocks."""
        code_data = block.get("code", {})
        text_content = code_data.get("rich_text", [])
        code = self._extract_rich_text(text_content)
        language = code_data.get("language", "")

        if code:
            return f"```{language}\n{code}\n```"
        return ""

    def _handle_quote(self, block: Dict[str, Any]) -> str:
        """Handle quote blocks."""
        text_content = block.get("quote", {}).get("rich_text", [])
        text = self._extract_rich_text(text_content)
        return f"> {text}" if text else ""

    def _handle_callout(self, block: Dict[str, Any]) -> str:
        """Handle callout blocks."""
        callout_data = block.get("callout", {})
        text_content = callout_data.get("rich_text", [])
        text = self._extract_rich_text(text_content)
        icon = callout_data.get("icon", {})

        icon_text = ""
        if icon.get("type") == "emoji":
            icon_text = icon.get("emoji", "")

        if text:
            return f"{icon_text} {text}" if icon_text else text
        return ""

    def _handle_divider(self, block: Dict[str, Any]) -> str:
        """Handle divider blocks."""
        return "---"

    def _handle_table(self, block: Dict[str, Any]) -> str:
        """Handle table blocks."""
        # Tables are complex and their rows are separate blocks
        # This is just a placeholder - actual table content comes from table_row blocks
        table_data = block.get("table", {})
        width = table_data.get("table_width", 0)
        logger.debug(f"Processing table block with width: {width}")
        return "[Table]"

    def _handle_table_row(self, block: Dict[str, Any]) -> str:
        """Handle table row blocks."""
        cells = block.get("table_row", {}).get("cells", [])
        row_texts = []
        for cell in cells:
            cell_text = self._extract_rich_text(cell)
            row_texts.append(cell_text)
        return " | ".join(row_texts)

    def _handle_column(self, block: Dict[str, Any]) -> str:
        """Handle column blocks."""
        # Columns are containers, their content is in child blocks
        # Log the block type for debugging
        logger.debug(f"Processing column block with ID: {block.get('id', 'unknown')}")
        return ""

    def _handle_column_list(self, block: Dict[str, Any]) -> str:
        """Handle column list blocks."""
        # Column lists are containers, their content is in child blocks
        # Log the block type for debugging
        logger.debug(
            f"Processing column_list block with ID: {block.get('id', 'unknown')}"
        )
        return ""

    def _handle_link_to_page(self, block: Dict[str, Any]) -> str:
        """Handle link to page blocks."""
        link_data = block.get("link_to_page", {})
        if link_data.get("type") == "page_id":
            return f"[Link to page: {link_data.get('page_id', '')}]"
        return "[Link to page]"

    def _handle_image(self, block: Dict[str, Any]) -> str:
        """Handle image blocks."""
        image_data = block.get("image", {})
        caption = image_data.get("caption", [])
        caption_text = self._extract_rich_text(caption)

        if caption_text:
            return f"[Image: {caption_text}]"
        return "[Image]"

    def _handle_video(self, block: Dict[str, Any]) -> str:
        """Handle video blocks."""
        video_data = block.get("video", {})
        caption = video_data.get("caption", [])
        caption_text = self._extract_rich_text(caption)

        if caption_text:
            return f"[Video: {caption_text}]"
        return "[Video]"

    def _handle_file(self, block: Dict[str, Any]) -> str:
        """Handle file blocks."""
        file_data = block.get("file", {})
        caption = file_data.get("caption", [])
        caption_text = self._extract_rich_text(caption)

        if caption_text:
            return f"[File: {caption_text}]"
        return "[File]"

    def _handle_pdf(self, block: Dict[str, Any]) -> str:
        """Handle PDF blocks."""
        pdf_data = block.get("pdf", {})
        caption = pdf_data.get("caption", [])
        caption_text = self._extract_rich_text(caption)

        if caption_text:
            return f"[PDF: {caption_text}]"
        return "[PDF]"

    def _handle_bookmark(self, block: Dict[str, Any]) -> str:
        """Handle bookmark blocks."""
        bookmark_data = block.get("bookmark", {})
        url = bookmark_data.get("url", "")
        caption = bookmark_data.get("caption", [])
        caption_text = self._extract_rich_text(caption)

        if caption_text:
            return f"[Bookmark: {caption_text} - {url}]"
        elif url:
            return f"[Bookmark: {url}]"
        return "[Bookmark]"

    def _handle_embed(self, block: Dict[str, Any]) -> str:
        """Handle embed blocks."""
        embed_data = block.get("embed", {})
        url = embed_data.get("url", "")

        if url:
            return f"[Embed: {url}]"
        return "[Embed]"
