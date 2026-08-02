"""Minimal file and beat-event output used by Spectrum."""

import contextlib
import io

import numpy as np


ENCODING = 'utf8'


@contextlib.contextmanager
def open_file(filename, mode='r'):
    """Yield a path or existing file handle and close only paths we open."""
    if isinstance(filename, str):
        handle = opened = io.open(filename, mode)
    else:
        handle = filename
        opened = None
    try:
        yield handle
    finally:
        if opened is not None:
            opened.close()


def write_events(events, filename, fmt='%.3f', delimiter='\t', header=None,
                 prefix=None):
    """Write one formatted event per line."""
    prefix = prefix or ''
    if isinstance(fmt, (list, tuple)):
        fmt = delimiter.join(fmt)
    with open_file(filename, 'wb') as handle:
        if header is not None:
            handle.write(('# ' + header + '\n').encode(ENCODING))
        for event in np.array(events):
            try:
                value = fmt % tuple(event.tolist())
            except AttributeError:
                value = event
            except TypeError:
                value = fmt % event
            handle.write((prefix + value + '\n').encode(ENCODING))
            handle.flush()


def write_beats(beats, filename, fmt=None, delimiter='\t', header=None,
                prefix=None):
    """Write detected beat timestamps."""
    if fmt is None and beats.ndim == 2:
        fmt = ['%.3f', '%d']
    elif fmt is None:
        fmt = '%.3f'
    write_events(beats, filename, fmt, delimiter, header, prefix=prefix)
