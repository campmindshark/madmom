===============================
Spectrum's Madmom beat tracker
===============================

This repository is a private, application-specific fork of `Madmom
<https://github.com/CPJKU/madmom>`_. It contains only Spectrum's online
RNN/DBN beat-tracking runtime. It is not a general-purpose Madmom distribution.

Supported boundary
==================

Spectrum launches the single installed program with a named PortAudio host API::

    DBNBeatTracker --host_api_name auto --audio_input=0 online

The tracker can instead consume mono or interleaved signed-16-bit
little-endian PCM from stdin::

    DBNBeatTracker --pcm_stdin online

An optional input audio file is accepted only for deterministic build and
relocated-runtime verification. Detected timestamps are emitted as ``BEAT:``
lines. Offline BLSTM, TCN, MIDI, evaluation, batch, pickle, and unrelated MIR
surfaces are intentionally absent.

Development
===========

The supported runtime is CPython 3.11 x64. The Windows build creates an
isolated environment, compiles the HMM and neural-network layer extensions,
builds a wheel, runs Spectrum's focused boundary tests, tests a fresh wheel,
and can stage a relocatable runtime::

    .\scripts\build.ps1

After the initial build, rerun the focused suite with::

    .\scripts\test.ps1

Linux uses ``scripts/build.sh`` and additionally verifies raw PCM stdin and a
copied relocatable runtime.

Licensing and attribution
=========================

The retained Madmom source is licensed under the BSD 3-Clause license in
``LICENSE``. The retained pretrained model files have their own license in
``madmom/models/LICENSE`` and are distributed under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0. The original work is by the
Department of Computational Perception at Johannes Kepler University Linz and
the Austrian Research Institute for Artificial Intelligence.
