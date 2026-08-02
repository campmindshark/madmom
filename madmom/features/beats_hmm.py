"""HMM state, transition, and observation models for online beats."""

import numpy as np

from madmom.ml.hmm import ObservationModel, TransitionModel


class BeatStateSpace:
    """Discretized beat positions for each modeled tempo interval."""

    def __init__(self, min_interval, max_interval, num_intervals=None):
        intervals = np.arange(np.round(min_interval),
                              np.round(max_interval) + 1)
        if num_intervals is not None and num_intervals < len(intervals):
            num_log_intervals = num_intervals
            intervals = []
            while len(intervals) < num_intervals:
                intervals = np.logspace(
                    np.log2(min_interval),
                    np.log2(max_interval),
                    num_log_intervals,
                    base=2,
                )
                intervals = np.unique(np.round(intervals))
                num_log_intervals += 1
        self.intervals = np.ascontiguousarray(intervals, dtype=int)
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)
        self.first_states = np.cumsum(
            np.r_[0, self.intervals[:-1]]).astype(int)
        self.last_states = np.cumsum(self.intervals) - 1
        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=int)
        index = 0
        for interval in self.intervals:
            self.state_positions[index:index + interval] = np.linspace(
                0, 1, interval, endpoint=False)
            self.state_intervals[index:index + interval] = interval
            index += interval


def exponential_transition(from_intervals, to_intervals, transition_lambda,
                           threshold=np.spacing(1), norm=True):
    """Return exponential tempo-transition probabilities."""
    if transition_lambda is None:
        return np.diag(np.diag(np.ones((len(from_intervals),
                                        len(to_intervals)))))
    ratio = (to_intervals.astype(float) /
             from_intervals.astype(float)[:, np.newaxis])
    probabilities = np.exp(-transition_lambda * abs(ratio - 1.))
    probabilities[probabilities <= threshold] = 0
    if norm:
        probabilities /= np.sum(probabilities, axis=1)[:, np.newaxis]
    return probabilities


class BeatTransitionModel(TransitionModel):
    """Allow tempo changes only at beat boundaries."""

    def __init__(self, state_space, transition_lambda):
        self.state_space = state_space
        self.transition_lambda = float(transition_lambda)
        states = np.arange(state_space.num_states, dtype=np.uint32)
        states = np.setdiff1d(states, state_space.first_states)
        previous_states = states - 1
        probabilities = np.ones_like(states, dtype=float)

        to_states = state_space.first_states
        from_states = state_space.last_states
        from_intervals = state_space.state_intervals[from_states]
        to_intervals = state_space.state_intervals[to_states]
        transition_probabilities = exponential_transition(
            from_intervals, to_intervals, self.transition_lambda)
        from_probability, to_probability = np.nonzero(
            transition_probabilities)
        states = np.hstack((states, to_states[to_probability]))
        previous_states = np.hstack(
            (previous_states, from_states[from_probability]))
        probabilities = np.hstack(
            (probabilities, transition_probabilities[
                transition_probabilities != 0]))
        transitions = self.make_sparse(
            states, previous_states, probabilities)
        super().__init__(*transitions)


class RNNBeatTrackingObservationModel(ObservationModel):
    """Map RNN activations to beat and non-beat HMM observations."""

    def __init__(self, state_space, observation_lambda):
        self.observation_lambda = observation_lambda
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        pointers[state_space.state_positions < 1. / observation_lambda] = 1
        super().__init__(pointers)

    def log_densities(self, observations):
        observations = np.array(
            observations, copy=False, subok=True, ndmin=1)
        densities = np.empty((len(observations), 2), dtype=float)
        densities[:, 0] = np.log(
            (1. - observations) / (self.observation_lambda - 1))
        densities[:, 1] = np.log(observations)
        return densities
