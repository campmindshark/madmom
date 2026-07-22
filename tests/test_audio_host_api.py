# encoding: utf-8
"""Tests for Spectrum's portable live-audio host selection."""

from __future__ import absolute_import, division, print_function

import importlib.util
import pathlib
import unittest


MODULE_PATH = (pathlib.Path(__file__).parent.parent / 'madmom' / 'audio' /
               'host_api.py')
SPEC = importlib.util.spec_from_file_location(
    'spectrum_madmom_host_api', MODULE_PATH)
HOST_API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST_API)
resolve_host_api_index = HOST_API.resolve_host_api_index


class FakePyAudio(object):
    """Small PyAudio substitute which never opens an audio device."""

    def __init__(self, names):
        self.names = names
        self.terminated = False

    def get_host_api_count(self):
        return len(self.names)

    def get_host_api_info_by_index(self, index):
        return {'name': self.names[index]}

    def terminate(self):
        self.terminated = True


class HostApiTests(unittest.TestCase):

    def test_explicit_host_api_name_is_case_insensitive(self):
        audio = FakePyAudio(['MME', 'Windows WASAPI'])
        self.assertEqual(resolve_host_api_index('wasapi', audio), 1)
        self.assertTrue(audio.terminated)

    def test_auto_uses_platform_preference_not_enumeration_order(self):
        audio = FakePyAudio(['JACK Audio Connection Kit', 'ALSA'])
        self.assertEqual(resolve_host_api_index(
            'auto', audio, system_name='Linux'), 1)
        self.assertTrue(audio.terminated)

    def test_missing_host_api_reports_available_names(self):
        audio = FakePyAudio(['ALSA'])
        with self.assertRaisesRegex(
                RuntimeError, r'WASAPI.*available: ALSA'):
            resolve_host_api_index('WASAPI', audio)
        self.assertTrue(audio.terminated)


if __name__ == '__main__':
    unittest.main()
