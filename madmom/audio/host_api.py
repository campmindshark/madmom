# encoding: utf-8
"""PortAudio host-API selection for Spectrum's live beat tracker."""

from __future__ import absolute_import, division, print_function

import platform


_AUTO_HOST_APIS = {
    'windows': ('WASAPI',),
    'linux': ('ALSA', 'JACK', 'PULSE'),
    'darwin': ('CORE AUDIO',),
}


def resolve_host_api_index(requested_name, audio_instance=None,
                           system_name=None):
    """Return the PortAudio host-API index matching ``requested_name``.

    ``auto`` uses a platform-specific preference order. The optional instance
    and system name make the policy testable without opening audio hardware.
    """
    if not requested_name or not str(requested_name).strip():
        raise RuntimeError('an audio host API name is required')

    if audio_instance is None:
        try:
            import pyaudio
        except ImportError as exc:
            raise RuntimeError('--host_api_name requires PyAudio') from exc
        audio_instance = pyaudio.PyAudio()

    try:
        host_apis = []
        for index in range(audio_instance.get_host_api_count()):
            info = audio_instance.get_host_api_info_by_index(index)
            host_apis.append((index, str(info['name'])))

        requested = str(requested_name).strip()
        if requested.casefold() == 'auto':
            current_system = system_name or platform.system()
            selectors = _AUTO_HOST_APIS.get(
                current_system.casefold(), ())
        else:
            selectors = (requested,)

        for selector in selectors:
            normalized_selector = selector.casefold()
            for index, name in host_apis:
                if name.casefold() == normalized_selector:
                    return index
            for index, name in host_apis:
                if normalized_selector in name.casefold():
                    return index

        available = ', '.join(name for _, name in host_apis) or 'none'
        raise RuntimeError(
            '%s host API not found (available: %s)' %
            (requested, available))
    finally:
        audio_instance.terminate()
