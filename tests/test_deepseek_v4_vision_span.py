"""Span-bidirectional attention: port math vs the DeepSeek reference.

Validates, without a GPU:
- compute_image_visibility (AST-extracted from vision_model.py) against the
  reference ``get_image_visible`` from the Vision-Exp checkpoint repo
  (embedded below as the oracle), on batched [1, seq] layouts.
- the SWA range arithmetic of the patched ROCm combine kernel (replicated in
  python) against the reference ``get_window_topk_idxs_visible`` row sets.
- exact causal equivalence for text-only batches.

Needs torch; skipped where unavailable (run inside the vllm container).
"""

import ast
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
VISION_MODEL = (
    ROOT
    / "container/rootfs/opt/venv/lib/python3.12/site-packages/vllm/models"
    / "deepseek_v4/vision_model.py"
)
ROCM = ROOT  # kernel arithmetic replicated inline; source checked textually

VOCAB = 129280
MAX_IMG = 384
WIN = 128
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)


def load_compute_image_visibility():
    tree = ast.parse(VISION_MODEL.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_image_visibility"
    ]
    ns = {"torch": torch, "IMAGE_START": IMAGE_START, "IMAGE_END": IMAGE_END}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(VISION_MODEL), "exec"), ns)
    return ns["compute_image_visibility"]


# --- reference oracle (DeepSeek-V4-Flash-Vision-Exp inference/model.py) -----

def ref_get_image_visible(input_ids, vocab_size, max_image_tokens):
    seqlen = input_ids.size(1)
    idx = torch.arange(seqlen, dtype=torch.int32).unsqueeze(0)
    is_start = input_ids == vocab_size + IMAGE_START
    is_end = input_ids == vocab_size + IMAGE_END
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(1)[0]
    left = (idx - starts) * valid
    ends = torch.where(is_end, idx, seqlen).flip(1).cummin(1)[0].flip(1)
    right = (ends - idx) * valid
    return left.clamp(max=max_image_tokens - 1), right.clamp(max=max_image_tokens)


def ref_get_window_topk_idxs_visible(window_size, seqlen, left, right, max_image_tokens):
    width = min(seqlen, window_size + max_image_tokens)
    idx = torch.arange(seqlen).unsqueeze(0)
    left_add = (left - (window_size - 1)).clamp(min=0)
    starts = (idx - (window_size - 1) - left_add).clamp(min=0)
    matrix = starts.unsqueeze(-1) + torch.arange(width)
    matrix = torch.where(matrix > (idx + right).unsqueeze(-1), -1, matrix)
    return matrix.int().contiguous()


# --- replicated kernel arithmetic (amd/rocm.py combine kernel) --------------

def kernel_swa_range(pos, vleft, vright, win, seq_len, gather_start, swa_width):
    left_add = max(vleft - (win - 1), 0)
    swa_start = max(pos - (win - 1) - left_add, 0, gather_start)
    swa_end = min(pos + vright, seq_len - 1)
    swa_len = min(swa_end - swa_start + 1, swa_width)
    return swa_start, swa_len


def kernel_swa_range_causal(pos, win):
    swa_len = min(pos + 1, win)
    return pos - swa_len + 1, swa_len


def make_span(n_img, n_newlines=2, n_pads=2):
    return (
        [VOCAB + IMAGE_START]
        + [VOCAB + IMAGE_PAD] * n_pads
        + [VOCAB + IMAGE] * n_img
        + [VOCAB + IMAGE_NEW_LINE] * n_newlines
        + [VOCAB + IMAGE_END]
    )


@unittest.skipUnless(torch is not None, "torch required")
class SpanVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.compute = load_compute_image_visibility()

    def _ids(self, layout):
        return torch.tensor(layout, dtype=torch.int64)

    def test_matches_reference_on_intact_spans(self):
        g = torch.Generator().manual_seed(7)
        for _ in range(25):
            seq = []
            for _ in range(int(torch.randint(1, 4, (1,), generator=g))):
                seq += torch.randint(0, VOCAB, (int(torch.randint(1, 60, (1,), generator=g)),), generator=g).tolist()
                seq += make_span(int(torch.randint(1, MAX_IMG + 1, (1,), generator=g)))
            seq += torch.randint(0, VOCAB, (30,), generator=g).tolist()
            ids = self._ids(seq)
            left, right = self.compute(ids, VOCAB, MAX_IMG)
            rl, rr = ref_get_image_visible(ids.unsqueeze(0), VOCAB, MAX_IMG)
            torch.testing.assert_close(left, rl[0].to(torch.int32))
            torch.testing.assert_close(right, rr[0].to(torch.int32))

    def test_text_only_is_all_zero(self):
        ids = self._ids(list(range(500)))
        left, right = self.compute(ids, VOCAB, MAX_IMG)
        self.assertEqual(int(left.abs().sum()), 0)
        self.assertEqual(int(right.abs().sum()), 0)

    def test_truncated_fragments_are_causal(self):
        span = make_span(16)
        # END lost to the next chunk
        headless = self._ids([1, 2] + span[:-3])
        left, right = self.compute(headless, VOCAB, MAX_IMG)
        self.assertEqual(int(left.abs().sum()), 0)
        self.assertEqual(int(right.abs().sum()), 0)
        # START lost to the previous chunk
        tailless = self._ids(span[3:] + [1, 2])
        left, right = self.compute(tailless, VOCAB, MAX_IMG)
        self.assertEqual(int(left.abs().sum()), 0)
        self.assertEqual(int(right.abs().sum()), 0)

    def test_kernel_ranges_match_reference_rows(self):
        seq = (
            list(range(50))
            + make_span(40)
            + list(range(60))
            + make_span(200)
            + list(range(40))
        )
        ids = self._ids(seq)
        n = len(seq)
        left, right = self.compute(ids, VOCAB, MAX_IMG)
        ref = ref_get_window_topk_idxs_visible(
            WIN, n, left.unsqueeze(0).to(torch.int64),
            right.unsqueeze(0).to(torch.int64), MAX_IMG,
        )[0]
        for pos in range(n):
            expected = set(ref[pos][ref[pos] >= 0].tolist())
            start, ln = kernel_swa_range(
                pos, int(left[pos]), int(right[pos]),
                WIN, seq_len=n, gather_start=0, swa_width=WIN + MAX_IMG,
            )
            self.assertEqual(set(range(start, start + ln)), expected, f"pos={pos}")

    def test_kernel_causal_path_unchanged_for_text(self):
        for pos in range(300):
            vis = kernel_swa_range(pos, 0, 0, WIN, 10_000, 0, WIN + MAX_IMG)
            causal = kernel_swa_range_causal(pos, WIN)
            self.assertEqual(vis, causal)

    def test_kernel_range_never_exceeds_buffer(self):
        # width bound: swa_len <= WIN + MAX_IMG for any clamped (left, right)
        for vleft in (0, WIN - 1, MAX_IMG - 1):
            for vright in (0, 1, MAX_IMG):
                _, ln = kernel_swa_range(
                    5_000, vleft, vright, WIN, 1_000_000, 0, WIN + MAX_IMG
                )
                self.assertLessEqual(ln, WIN + MAX_IMG)


if __name__ == "__main__":
    unittest.main()
