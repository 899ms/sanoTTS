"""ATen-compatible MT19937 + normal_fill_16, in numpy.

The nano decoder is noise-fed, so a rendering is only reproducible if the
noise stream is. PyTorch's is MT19937 seeded from the low 32 bits, a 24-bit
uniform, and Box-Muller in blocks of 16 -- reproduced here so that Python,
the C runtime and PyTorch all render a given seed the same way.

`numpy.random.RandomState` is also MT19937, but its seeding, its tempering
draw order and its Gaussian transform all differ from ATen's, so it cannot be
substituted. The arithmetic below is the ATen path exactly.

Exactness, measured rather than assumed: the integer half is bit-exact and
`mcu/test/fixtures/en_us_e12nano/e2e_uniform.bin` proves it. The Gaussian half
goes through log/cos/sin, which are not bit-identical across libm builds, so a
minority of draws differ by a few ulp -- the C runtime documents the same
limit and the fixture bounds it at 1e-5.
"""

from __future__ import annotations

import numpy as np

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF


class ATenMT19937:
    """MT19937 with ATen's seeding and 24-bit float uniform."""

    __slots__ = ("_state", "_left", "_next")

    def __init__(self, seed: int) -> None:
        state = np.empty(_N, dtype=np.uint64)
        state[0] = np.uint64(seed & 0xFFFFFFFF)
        for i in range(1, _N):
            prev = int(state[i - 1])
            state[i] = np.uint64((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
        self._state = state.astype(np.uint32)
        self._left = 1
        self._next = 0

    def _next_state(self) -> None:
        s = self._state.astype(np.int64)
        self._left = _N
        self._next = 0
        for i in range(_N):
            u = int(s[i])
            v = int(s[(i + 1) % _N])
            mixed = (u & _UPPER) | (v & _LOWER)
            twist = (mixed >> 1) ^ (_MATRIX_A if (v & 1) else 0)
            s[i] = int(s[(i + _M) % _N]) ^ twist
        self._state = (s & 0xFFFFFFFF).astype(np.uint32)

    def next_uint32(self) -> int:
        self._left -= 1
        if self._left <= 0:
            self._next_state()
        y = int(self._state[self._next])
        self._next += 1
        y ^= y >> 11
        y = (y ^ ((y << 7) & 0x9D2C5680)) & 0xFFFFFFFF
        y = (y ^ ((y << 15) & 0xEFC60000)) & 0xFFFFFFFF
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def uniform(self, n: int) -> np.ndarray:
        """at::uniform_real_distribution<float>: (raw & (2**24 - 1)) * 2**-24."""
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = np.float32((self.next_uint32() & 0xFFFFFF) * (1.0 / 16777216.0))
        return out


def _normal_fill_16(block: np.ndarray) -> None:
    """ATen normal_fill_16, mean 0 std 1, float32 throughout. In place."""
    u1 = np.float32(1.0) - block[:8]
    u2 = block[8:]
    radius = np.sqrt(np.float32(-2.0) * np.log(u1, dtype=np.float32), dtype=np.float32)
    theta = np.float32(2.0 * np.pi) * u2
    block[:8] = radius * np.cos(theta, dtype=np.float32)
    block[8:] = radius * np.sin(theta, dtype=np.float32)


def uniform_stream(seed: int, n: int) -> np.ndarray:
    return ATenMT19937(seed).uniform(n)


def seeded_noise(seed: int, channels: int, frames: int) -> np.ndarray:
    """[channels, frames] of N(0,1), matching torch.randn under this seed.

    Torch sends sizes under 16 to a scalar path this does not implement; the
    decoder always asks for channels*frames well above that.
    """
    size = channels * frames
    if size < 16:
        raise ValueError(f"seeded_noise needs at least 16 values, got {size}")
    gen = ATenMT19937(seed)
    flat = gen.uniform(size)
    whole = (size // 16) * 16
    for i in range(0, whole, 16):
        _normal_fill_16(flat[i : i + 16])
    if size % 16:
        # Torch draws a FRESH block of 16 (continuing the same stream) and
        # overwrites the last 16 values with it. The remainder left by the
        # loop above is discarded, so the tail is not simply its leftover.
        tail = gen.uniform(16)
        _normal_fill_16(tail)
        flat[size - 16 :] = tail
    return flat.reshape(channels, frames)
