"""Spectrum-owned tests for the private online DBN beat tracker."""

import io
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys

import numpy as np
import pytest

from madmom.audio.host_api import resolve_host_api_index
from madmom.audio.signal import RawPcmStream
from madmom.io import write_beats


ROOT = Path(__file__).resolve().parent.parent
TRACKER = ROOT / 'bin' / 'DBNBeatTracker'
SAMPLE = ROOT / 'tests' / 'data' / 'audio' / 'sample.wav'
EXPECTED_BEATS = ['BEAT:0.470', 'BEAT:0.790', 'BEAT:1.480',
                  'BEAT:2.160', 'BEAT:2.500']


class FakePyAudio:
    """PyAudio substitute that exposes host APIs without opening a device."""

    def __init__(self, names):
        self.names = names
        self.terminated = False

    def get_host_api_count(self):
        return len(self.names)

    def get_host_api_info_by_index(self, index):
        return {'name': self.names[index]}

    def terminate(self):
        self.terminated = True


def test_supported_command_line_contract_is_online_only():
    tracker = runpy.run_path(str(TRACKER))
    parser = tracker['create_parser']()

    args = parser.parse_args([
        '--host_api_name', 'auto', '--audio_input=7', 'online'])

    assert args.host_api_name == 'auto'
    assert args.audio_input == 7
    assert args.mode == 'online'
    for removed in ('single', 'batch', 'pickle'):
        with pytest.raises(SystemExit):
            parser.parse_args([removed])
    with pytest.raises(SystemExit):
        parser.parse_args(['--host_api'])


def test_explicit_host_api_name_is_case_insensitive():
    audio = FakePyAudio(['MME', 'Windows WASAPI'])

    assert resolve_host_api_index('wasapi', audio) == 1
    assert audio.terminated


def test_auto_uses_platform_preference_not_enumeration_order():
    audio = FakePyAudio(['JACK Audio Connection Kit', 'ALSA'])

    assert resolve_host_api_index('auto', audio, system_name='Linux') == 1
    assert audio.terminated


def test_missing_host_api_reports_available_names():
    audio = FakePyAudio(['ALSA'])

    with pytest.raises(RuntimeError, match=r'WASAPI.*available: ALSA'):
        resolve_host_api_index('WASAPI', audio)
    assert audio.terminated


def test_mono_pcm_is_normalized_and_framed():
    raw = struct.pack('<hhhh', -32768, -16384, 0, 32767)
    stream = RawPcmStream(io.BytesIO(raw), sample_rate=4,
                          num_channels=1, frame_size=4, hop_size=2)

    first = next(stream).copy()
    second = next(stream)

    assert np.allclose(first, [0., 0., -1., -0.5])
    assert np.allclose(second, [-1., -0.5, 0., 32767. / 32768.])
    assert first.start == 0.
    assert second.start == 0.5
    with pytest.raises(StopIteration):
        next(stream)


def test_stereo_pcm_is_mixed_to_mono():
    raw = struct.pack('<hhhh', 32767, -32767, -32768, -32768)
    stream = RawPcmStream(io.BytesIO(raw), sample_rate=2,
                          num_channels=2, frame_size=2, hop_size=2)

    assert np.allclose(next(stream), [0., -1.])


def test_beat_output_uses_spectrum_event_prefix():
    output = io.BytesIO()

    write_beats(np.array([0.1]), output, prefix='BEAT:')

    assert output.getvalue() == b'BEAT:0.100\n'


def test_online_rnn_dbn_pipeline_emits_expected_beats():
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(
        filter(None, (str(ROOT), environment.get('PYTHONPATH'))))

    result = subprocess.run(
        [sys.executable, str(TRACKER), 'online', str(SAMPLE)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == EXPECTED_BEATS
