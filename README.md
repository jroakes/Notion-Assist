# Notion Assist

An intelligent agent that processes Notion tickets, executes tasks with various tools, and verifies completion.

## Features

- Extract content from Notion pages via URL
- LangGraph orchestration for agent workflows
- Tool integrations: Google Analytics, Search Console, web crawling
- Streamlit UI with approval workflow
- Automated task verification

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Add your NOTION_TOKEN to .env
```

3. Activate virtual environment:
```bash
source notionui/bin/activate
```

## Usage

Test Notion extraction:
```bash
python test/test_notion_url_extraction.py "https://notion.so/your-page-url"
```

## Project Structure

```
src/
├── extractors/       # Content extraction framework
│   ├── base.py      # Abstract base class
│   └── notion.py    # Notion API implementation
└── settings.py      # Configuration management

test/                # Test files
```

## Development

- Python 3.11+
- Follow code standards in CLAUDE.md
- Run tests: `python -m pytest test/`
- Format code: `black src/ test/`
- Lint: `flake8 src/ test/ --extend-ignore=E501`