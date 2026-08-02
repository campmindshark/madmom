"""Spectrum's online neural-network and DBN beat-processing pipeline."""

import time

import numpy as np

from ..ml.nn import average_predictions
from ..processors import ParallelProcessor, Processor, SequentialProcessor


class RNNBeatProcessor(SequentialProcessor):
    """Produce online beat activations with the 2016 LSTM ensemble."""

    def __init__(self, post_processor=average_predictions, online=True,
                 nn_files=None, **kwargs):
        from ..audio.signal import SignalProcessor, FramedSignalProcessor
        from ..audio.stft import ShortTimeFourierTransformProcessor
        from ..audio.spectrogram import (
            FilteredSpectrogramProcessor,
            LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor,
        )
        from ..ml.nn import NeuralNetworkEnsemble
        from ..models import BEATS_LSTM

        if not online:
            raise ValueError('Spectrum supports only online RNN beat tracking')
        if nn_files is None:
            nn_files = BEATS_LSTM

        signal = SignalProcessor(num_channels=1, sample_rate=44100)
        frames = FramedSignalProcessor(frame_size=2048, **kwargs)
        stft = ShortTimeFourierTransformProcessor()
        filtered = FilteredSpectrogramProcessor(
            num_bands=12,
            fmin=30,
            fmax=17000,
            norm_filters=True,
        )
        logarithmic = LogarithmicSpectrogramProcessor(mul=1, add=1)
        difference = SpectrogramDifferenceProcessor(
            diff_ratio=0.5,
            positive_diffs=True,
            stack_diffs=np.hstack,
        )
        spectral = SequentialProcessor(
            (frames, stft, filtered, logarithmic, difference))
        features = ParallelProcessor([spectral])
        pre_processor = SequentialProcessor((signal, features, np.hstack))
        network = NeuralNetworkEnsemble.load(
            nn_files,
            ensemble_fn=post_processor,
            **kwargs,
        )
        super().__init__((pre_processor, network))


class DBNBeatTrackingProcessor(Processor):
    """Decode online beat activations with Spectrum's DBN/HMM."""

    MIN_BPM = 55.
    MAX_BPM = 215.
    NUM_TEMPI = None
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16

    def __init__(self, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI,
                 transition_lambda=TRANSITION_LAMBDA,
                 observation_lambda=OBSERVATION_LAMBDA,
                 fps=None, online=True, **kwargs):
        from .beats_hmm import (
            BeatStateSpace,
            BeatTransitionModel,
            RNNBeatTrackingObservationModel,
        )
        from ..ml.hmm import HiddenMarkovModel

        if not online:
            raise ValueError('Spectrum supports only online DBN beat tracking')
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        self.st = BeatStateSpace(min_interval, max_interval, num_tempi)
        self.tm = BeatTransitionModel(self.st, transition_lambda)
        self.om = RNNBeatTrackingObservationModel(
            self.st, observation_lambda)
        self.hmm = HiddenMarkovModel(self.tm, self.om, None)
        self.fps = fps
        self.max_bpm = max_bpm
        self.counter = 0
        self.last_beat = 0
        self.absolute_time = kwargs.get('absolute_time', False)
        self.ticktime = None

    def reset(self):
        """Reset all state used by the online decoder."""
        self.hmm.reset()
        self.counter = 0
        self.last_beat = 0
        self.ticktime = time.monotonic()

    def process(self, activations, reset=True, **kwargs):
        """Return beat timestamps detected in one or more activation frames."""
        if not isinstance(activations, np.ndarray):
            activations = np.array(activations, ndmin=1)
        if reset:
            self.reset()

        forward = self.hmm.forward(activations, reset=reset)
        states = np.argmax(forward, axis=1)
        beat_states = self.om.pointers[states] == 1

        beats = []
        for frame in np.nonzero(beat_states)[0]:
            current = (frame + self.counter) / float(self.fps)
            if current >= self.last_beat + 60. / self.max_bpm:
                self.last_beat = current
                beats.append(current)
        self.counter += len(activations)

        result = np.array(beats)
        if self.absolute_time:
            if self.ticktime is None:
                self.ticktime = time.monotonic()
            result = result + self.ticktime
        return result

    @staticmethod
    def add_arguments(parser, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                      num_tempi=NUM_TEMPI,
                      transition_lambda=TRANSITION_LAMBDA,
                      observation_lambda=OBSERVATION_LAMBDA):
        """Add the online DBN settings supported by Spectrum."""
        group = parser.add_argument_group(
            'dynamic Bayesian Network arguments')
        group.add_argument(
            '--min_bpm', type=float, default=min_bpm,
            help='minimum tempo [bpm, default=%(default).2f]')
        group.add_argument(
            '--max_bpm', type=float, default=max_bpm,
            help='maximum tempo [bpm, default=%(default).2f]')
        group.add_argument(
            '--num_tempi', type=int, default=num_tempi,
            help='limit modeled tempi and use logarithmic spacing')
        group.add_argument(
            '--transition_lambda', type=float, default=transition_lambda,
            help='tempo-transition lambda [default=%(default).1f]')
        group.add_argument(
            '--observation_lambda', type=float, default=observation_lambda,
            help='beat observation subdivisions [default=%(default)i]')
        return group
