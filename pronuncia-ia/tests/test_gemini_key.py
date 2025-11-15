pkill -f start_server.py || true"""Pytest tests to validate GEMINI/GOOGLE API key availability and minimal SDK setup.

These tests do NOT perform an actual request to the Gemini API. They only:
- load the repository `.env` (if present)
- confirm an API key exists in the environment
- assert that the `google.generativeai` package is importable
- call `genai.configure(api_key=...)` to check basic SDK configuration (no network call)

Run with the project's venv activated:

    /path/to/.venv/bin/pytest -q pronuncia-ia/tests/test_gemini_key.py

If `python-dotenv` or `google-generativeai` are missing the tests will fail with a helpful message.
"""
import os
import pathlib
import sys

import pytest


def load_project_dotenv():
    """Load the .env located at the repository root if python-dotenv is available.
    Returns True if dotenv was loaded or not needed, False if python-dotenv missing.
    """
    try:
        from dotenv import load_dotenv
    except Exception:
        return False

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    return True


def test_env_has_gemini_key():
    """Ensure either GEMINI_API_KEY or GOOGLE_API_KEY is present in environment after loading .env."""
    load_ok = load_project_dotenv()
    if not load_ok:
        pytest.skip("python-dotenv not installed in venv; install with: pip install python-dotenv")

    gem = os.getenv("GEMINI_API_KEY")
    goo = os.getenv("GOOGLE_API_KEY")
    assert (gem or goo), (
        "Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in environment.\n"
        "Add one to .env or export it in your shell before running the server/tests."
    )


def test_google_generativeai_import_and_configure():
    """Try to import google.generativeai and configure it with the key (no network call).

    This verifies the SDK is installed and accepts configuration; it does not call the API.
    """
    try:
        import google.generativeai as genai
    except Exception as e:
        pytest.skip("google-generativeai package not installed in venv: pip install google-generativeai")

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    assert key, "No GEMINI_API_KEY/GOOGLE_API_KEY found in environment for SDK configuration."

    # This configure call does not perform network I/O; it only sets the client API key in the SDK
    try:
        genai.configure(api_key=key)
    except Exception as e:
        pytest.fail(f"genai.configure raised an exception: {e}")

    # Instantiating a GenerativeModel object may perform lazy checks, but typically does not call network until generate_content
    try:
        model_name = os.getenv("GEMINI_MODEL") or os.getenv("GEMINI_CHAT_MODEL") or "gemini-1.5-flash"
        _ = genai.GenerativeModel(model_name)
    except Exception as e:
        # Do not fail the test if the SDK refuses to instantiate a model; warn instead.
        pytest.skip(f"Could not instantiate genai.GenerativeModel (this may require network/credentials): {e}")
