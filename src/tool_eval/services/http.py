"""Shared HTTP client construction."""

import httpx

USER_AGENT = "tool-eval/0.1 (research project; python-httpx)"


def build_client(*, timeout: float) -> httpx.Client:
    """Create an HTTP client with a descriptive User-Agent and a request timeout.

    Args:
        timeout: Timeout in seconds applied to each request phase.

    Returns:
        A configured `httpx.Client`; callers should use it as a context manager.
    """
    return httpx.Client(
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )
