"""
Configuration settings for Notion Assist application.

This module loads environment variables from .env file and provides
a centralized configuration object for use throughout the application.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv


class Config:
    """
    Configuration class that dynamically loads environment variables from .env file.
    """

    # Define required environment variables
    REQUIRED_VARS: List[str] = [
        "NOTION_TOKEN",
    ]

    # Define optional environment variables with defaults
    OPTIONAL_VARS: Dict[str, Any] = {
        # Example: 'API_TIMEOUT': 30,
        # Example: 'DEBUG': False,
    }

    def __init__(self) -> None:
        """Initialize Config by loading environment variables from .env file."""
        # Get the project root directory (parent of src)
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"

        # Load environment variables from .env file
        load_dotenv(env_path)

        # Load required variables
        self._load_required_vars()

        # Load optional variables with defaults
        self._load_optional_vars()

    def _load_required_vars(self) -> None:
        """Load required environment variables and raise error if missing."""
        missing_vars = []

        for var in self.REQUIRED_VARS:
            value = os.getenv(var)
            if value is None:
                missing_vars.append(var)
            else:
                setattr(self, var, value)

        if missing_vars:
            raise ValueError(
                f"Required environment variables not found: {', '.join(missing_vars)}. "
                "Please ensure they are set in your .env file."
            )

    def _load_optional_vars(self) -> None:
        """Load optional environment variables with defaults."""
        for var, default in self.OPTIONAL_VARS.items():
            value = os.getenv(var, default)
            # Handle boolean conversion
            if isinstance(default, bool) and isinstance(value, str):
                value = value.lower() in ("true", "1", "yes", "on")
            # Handle integer conversion
            elif isinstance(default, int) and isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    value = default
            setattr(self, var, value)

    def __getattr__(self, name: str) -> Any:
        """Allow dynamic access to any environment variable."""
        return os.getenv(name)

    def __repr__(self) -> str:
        """Return string representation of Config object."""
        attrs = []
        for var in self.REQUIRED_VARS:
            value = getattr(self, var, None)
            attrs.append(f"{var}={'***' if value else 'Not Set'}")
        for var in self.OPTIONAL_VARS:
            value = getattr(self, var, None)
            attrs.append(f"{var}={value}")
        return f"Config({', '.join(attrs)})"


# Create singleton instance
config = Config()
