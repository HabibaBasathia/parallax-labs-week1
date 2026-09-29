"""
Wraps calls to Groq's free LLM API with retry logic for rate limits
and timeouts, so a single blip doesn't crash the whole pipeline.
"""

import os
import time
from groq import Groq, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from .env

MODEL_NAME = "openai/gpt-oss-20b"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
MAX_OUTPUT_TOKENS = 512

_client = None


def get_client() -> Groq:
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
    Calls the LLM with retries on rate limits / connection errors.
    Returns the model's text response, or a clear error message string
    if all retries fail (never raises up to the caller).
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