"""Settings read from environment variables, with `.env` file support.

`load_dotenv()` copies KEY=value lines from a `.env` file (in the current
directory or any parent) into the process environment, without overriding
variables that are already set. Similar to Spring Boot reading
`application.properties`, but for secrets that must stay out of git.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or None


def openai_model(default: str) -> str:
    return os.environ.get("DOT2DOT_OPENAI_MODEL") or default
