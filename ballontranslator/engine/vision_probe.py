"""Synthetic image checks using the existing profile connection-test transport."""
from __future__ import annotations

import copy
import secrets

import cv2
import numpy as np

from ballontranslator.modules.llm_chat import (
    LLMConnectionError,
    _profile_connection_request,
    openai_chat_completion_args,
)
from ballontranslator.modules.llm_vision import encode_chat_image
from ballontranslator.utils.llm_profiles import LLMProfile, THINKING_DISABLED


def probe_profile_vision(profile: LLMProfile, *, proxy: str = '', timeout: float = 15.0) -> str:
    """Send one in-memory challenge, using the connection check's timeout.

    Connection failures propagate without changing the measured capability.
    Only the synthetic image contains the answer; the prompt cannot reveal it.

    >>> from unittest.mock import patch
    >>> with patch(__name__ + '._profile_connection_request') as request:
    ...     request.return_value.json.return_value = {'choices': []}
    ...     probe_profile_vision(LLMProfile(model='offline'))
    'no'
    """
    profile = copy.deepcopy(profile)
    if THINKING_DISABLED in profile.thinking_level_options:
        profile.thinking_level = THINKING_DISABLED
    model = profile.vision_model.strip() or profile.model.strip()
    digits = f'{secrets.randbelow(10000):04d}'
    image = np.full((96, 256, 3), 255, dtype=np.uint8)
    cv2.putText(image, digits, (20, 68), cv2.FONT_HERSHEY_SIMPLEX,
                2.5, (0, 0, 0), 4, cv2.LINE_AA)
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'Read the image. Reply with the four digits only.'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'What number is shown in this image?'},
                encode_chat_image(image, detail=profile.vision_detail_level).image_part(),
            ]},
        ],
        **openai_chat_completion_args(profile, model),
    }
    # The connection-test transport takes an HTTP body, not SDK extra_body.
    payload.update(payload.pop('extra_body', {}))
    try:
        response = _profile_connection_request(
            profile, 'chat/completions', proxy=proxy, timeout=timeout, payload=payload)
    except LLMConnectionError as error:
        # This transport exposes status through its own stable, sanitized prefix.
        # Other failures (auth, timeout, network, rate limit, server) remain errors.
        if error.kind == 'invalid_response' and str(error).startswith(
                ('HTTP 400:', 'HTTP 415:', 'HTTP 422:')):
            return 'no'
        raise
    try:
        content = response.json()['choices'][0]['message']['content']
    except (ValueError, KeyError, IndexError, TypeError):
        return 'no'
    return 'yes' if isinstance(content, str) and digits in content else 'no'
