r"""The mxfp4 decode's bit trick, checked against the e2m1 table. No GPU.

`container/native/ds4_moe_mxfp4.cpp` decodes an e2m1 nibble by placing its three
magnitude bits at (nib & 7) << 9, which lands e2m1's subnormal step on a genuine
fp16 subnormal, and folds the resulting 2^-14 back in through the per-block
scale. That is three ALU ops instead of about thirteen, and it is exact -- but
only for those exact shift constants. Change a 9 to an 8 and the kernel still
compiles, still runs, still looks plausible, and every weight is off by a factor
of two.

Reproducing the transform in python and checking it against the table the format
defines needs no GPU and no container, so it can run anywhere the rest of the
packaging tests do.

    python3 -m unittest discover -s tests -v
"""

import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = os.path.join(REPO, "container/native/ds4_moe_mxfp4.cpp")

# e2m1: sign at bit 3, exponent at bits 2:1, mantissa at bit 0.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
POST_SCALE = 16384.0          # 2^14, undoes the fp16 subnormal ladder


def fp16_bits_to_float(bits):
    """IEEE half from its 16 bits. Written out rather than pulled from numpy so
    the subnormal branch -- the whole point of the trick -- is visible here."""
    sign = -1.0 if bits >> 15 else 1.0
    exp = (bits >> 10) & 0x1F
    man = bits & 0x3FF
    if exp == 0:                       # subnormal: no implicit leading 1
        return sign * man * 2.0 ** -24
    if exp == 0x1F:
        return sign * float("inf") if man == 0 else float("nan")
    return sign * (1.0 + man / 1024.0) * 2.0 ** (exp - 15)


def kernel_decode(nib):
    """What fp4_to_f32_fp16 computes, times the scale the caller folds in."""
    bits = ((nib & 7) << 9) | ((nib & 8) << 12)
    return fp16_bits_to_float(bits) * POST_SCALE


class Mxfp4Decode(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(KERNEL):
            self.skipTest("MoE kernel source not present")

    def test_all_sixteen_codes_are_exact(self):
        for nib in range(16):
            want = E2M1[nib & 7] * (-1.0 if nib & 8 else 1.0)
            got = kernel_decode(nib)
            # exact equality, not almost-equal: every e2m1 magnitude is
            # representable in fp16 and in f32, and 2^14 is a power of two, so
            # there is no rounding anywhere in this path to be tolerant of.
            self.assertEqual(got, want,
                             f"nibble {nib:#x}: decoded {got!r}, table says {want!r}")

    def test_negative_zero_stays_zero(self):
        # nib 0x8 is -0. A sign bit smeared into the exponent shows up here first.
        self.assertEqual(abs(kernel_decode(0x8)), 0.0)

    def test_the_subnormal_step_is_what_makes_it_work(self):
        # em=1 -> 0.5 is the case the arithmetic decode needed a select for. It
        # is a genuine fp16 subnormal here; if fp16 subnormals were ever flushed
        # this is the value that would silently become 0.
        self.assertEqual((1 & 7) << 9, 0x0200)
        self.assertEqual(fp16_bits_to_float(0x0200) * POST_SCALE, 0.5)

    def test_source_still_spells_the_transform_this_way(self):
        with open(KERNEL) as f:
            src = f.read()
        self.assertIn("((nib & 7u) << 9) | ((nib & 8u) << 12)", src,
                      "the fp16 decode's shift constants changed; this test's "
                      "model of them is stale, so re-derive both together")
        # FP4_POST_SCALE is defined in both decode branches -- 1.0f for the
        # arithmetic reference, 2^14 for the fp16 path that is actually built.
        vals = re.findall(r"#define\s+FP4_POST_SCALE\s+([0-9.]+)f", src)
        self.assertEqual(len(vals), 2,
                         f"expected FP4_POST_SCALE in both decode branches, got {vals}")
        arith, default = (float(v) for v in vals)
        self.assertEqual(arith, 1.0)
        self.assertEqual(default, POST_SCALE)


if __name__ == "__main__":
    unittest.main()
