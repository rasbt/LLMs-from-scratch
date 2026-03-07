import os
import time
from functools import wraps

import requests


def _get_env_number(name, default, cast):
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        return cast(value)
    except ValueError:
        return default


def pytest_configure(config):
    if not getattr(config.option, "check_links", False):
        return

    timeout = _get_env_number("CHECK_LINKS_TIMEOUT", 10.0, float)
    max_retries = max(0, _get_env_number("CHECK_LINKS_RETRIES", 2, int))
    retry_backoff = max(0.0, _get_env_number("CHECK_LINKS_RETRY_BACKOFF", 1.0, float))
    current_request = requests.sessions.Session.request

    if getattr(current_request, "_check_links_wrapped", False):
        return

    retryable_methods = {"GET", "HEAD"}
    retryable_errors = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    )

    @wraps(current_request)
    def request_with_timeout(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout

        method_name = (method or "").upper()

        for attempt in range(max_retries + 1):
            try:
                return current_request(self, method, url, **kwargs)
            except retryable_errors:
                should_retry = method_name in retryable_methods and attempt < max_retries
                if not should_retry:
                    raise

                # Retries smooth over transient CI/network blips without masking real 4xx/5xx failures.
                if retry_backoff:
                    time.sleep(retry_backoff * (attempt + 1))

    request_with_timeout._check_links_wrapped = True
    requests.sessions.Session.request = request_with_timeout
