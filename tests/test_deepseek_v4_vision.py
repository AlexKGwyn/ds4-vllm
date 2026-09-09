"""Pure gates for the DeepSeek-V4 vision overlay.

These tests intentionally avoid importing torch/vLLM so they run on the host
and in CI before the 35 GB serving image is available.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
MM_PREPROCESS = (
    REPO
    / "container/rootfs/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/mm_preprocess.py"
)
VISION_DIR = MM_PREPROCESS.parent
PATCH = REPO / "container/patches/vllm-upstream.patch"


def patch_text() -> str:
    return PATCH.read_text()


def load_pure_preprocess() -> dict[str, object]:
    tree = ast.parse(MM_PREPROCESS.read_text())
    wanted = {
        "grid_tokens",
        "solve_resize_ratio",
        "safe_resize",
        "compress_pad_for_start",
        "image_token_count_for_size",
    }
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & {"COMPRESS_PAD_TO"}:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
    namespace: dict[str, object] = {"math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(MM_PREPROCESS), "exec"), namespace)
    return namespace


class TestDeepseekV4VisionPreprocessing(unittest.TestCase):
    def test_supplied_images_match_reference_token_counts(self) -> None:
        ns = load_pure_preprocess()
        count = ns["image_token_count_for_size"]
        self.assertEqual(count(1024, 701, 0), 313)  # carrots.jpeg
        # The 450x308 image is first raised to vision_min_pixels.
        self.assertEqual(count(450, 308, 0), 109)  # corn.jpeg

    def test_c4_leading_padding_depends_on_absolute_prompt_offset(self) -> None:
        ns = load_pure_preprocess()
        leading_pad = ns["compress_pad_for_start"]
        self.assertEqual([leading_pad(i) for i in range(8)], [3, 2, 1, 0, 3, 2, 1, 0])
        for start in range(16):
            self.assertEqual((start + leading_pad(start)) % 4, 3)

    def test_vision_tower_and_wrapper_are_packaged(self) -> None:
        self.assertTrue((VISION_DIR / "vision.py").is_file())
        self.assertTrue((VISION_DIR / "vision_model.py").is_file())
        wrapper = (VISION_DIR / "vision_model.py").read_text()
        self.assertIn("DeepseekV4VForConditionalGeneration", wrapper)
        self.assertIn("requires_raw_input_tokens = True", wrapper)
        self.assertIn("DeepseekV4VMultiModalProcessor", wrapper)

    def test_registry_and_input_validation_support_the_override_arch(self) -> None:
        patch = patch_text()
        self.assertIn('"DeepseekV4VForConditionalGeneration"', patch)
        self.assertIn("vision_n_layers", patch)
        self.assertIn("vocab_size + 4", patch)

    def test_amd_path_masks_sentinels_and_registers_visual_bias(self) -> None:
        patch = patch_text()
        self.assertIn("self.gate.bias_vl", patch)
        self.assertIn("input_ids >= self.vision_vocab_size", patch)
        self.assertIn("torch.zeros_like(input_ids)", patch)
        self.assertIn("vision_e_score_correction_bias", patch)
        self.assertIn('name.endswith(".ffn.gate.e_score_correction_bias")', patch)

    def test_visual_router_uses_dynamic_bias_without_changing_text_bias(self) -> None:
        patch = patch_text()
        self.assertIn("image_mask.unsqueeze(-1)", patch)
        self.assertIn("self.e_score_correction_bias", patch)
        self.assertIn("_topk_softplus_sqrt_torch", patch)


if __name__ == "__main__":
    unittest.main()
