"""
Wraps calls to Groq's free LLM API with retry logic for rate limits
and timeouts, so a single blip doesn't crash the whole pipeline.
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv
from groq import APIConnectionError, APIError, Groq, RateLimitError

load_dotenv()  # reads GROQ_API_KEY from .env

MODEL_NAME: str = "openai/gpt-oss-20b"
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: int = 5
MAX_OUTPUT_TOKENS: int = 512

_client: Optional[Groq] = None


def get_client() -> Groq:
    """
    Create the Groq client on first use, then reuse it.

    Raises:
        ValueError: If GROQ_API_KEY is missing from the .env file.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Make sure you have a .env file "
                "with GROQ_API_KEY=your_key_here in the project root."
            )
        _client = Groq(api_key=api_key)
    return _client


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Send a prompt to the LLM, retrying on rate limits and connection errors.

    Args:
        system_prompt: Instructions that tell the model how to behave.
        user_prompt: The actual question, including any context.

    Returns:
        The model's text answer. If all retries fail, a string starting
        with "Error:" is returned instead of raising, so callers can
        check for it.
    """
    client = get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.2,  # low temperature = more factual, less creative drift
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            print(f"[Rate limit hit] Attempt {attempt}/{MAX_RETRIES}, waiting {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)

        except APIConnectionError:
            print(f"[Connection error] Attempt {attempt}/{MAX_RETRIES}, waiting {RETRY_DELAY_SECONDS}s...")
            time.sleep(RETRY_DELAY_SECONDS)

        except APIError as e:
            print(f"[API error] {e}")
            return f"Error: the LLM API returned an error ({e})"

    return "Error: LLM call failed after multiple retries (rate limit or connection issue)."