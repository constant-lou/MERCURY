import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
INFER_DIR = ROOT / "src" / "infer"
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(INFER_DIR))
sys.path.insert(0, str(SRC_DIR))


class MultiViewReasoningTests(unittest.TestCase):
    def test_builds_semantic_roles_without_versioned_names(self):
        from multi_view_reasoning import build_multi_view_prompts

        sample = {
            "question": "What metaphor does this image convey?",
            "options": ["A literal answer", "A deeper answer"],
            "meta_data": {"language": "en"},
        }
        prompts = build_multi_view_prompts(sample, "en", "none")

        self.assertEqual(
            ["direct_anchor", "literal_challenge", "text_interaction"],
            [prompt["type"] for prompt in prompts],
        )
        self.assertEqual([2.0, 1.2, 1.0], [prompt["weight"] for prompt in prompts])


class EvidenceCalibrationTests(unittest.TestCase):
    def test_cvec_switches_to_complementary_candidate(self):
        from evidence_calibration import calibrate_cross_view_prediction

        sample = {
            "options": [
                "The literal surface description.",
                "The image criticizes unequal opportunity.",
            ]
        }
        baseline = {
            "final_prediction": "A",
            "view_logs": [
                {"type": "direct_anchor", "pred": "A", "weight": 1.0, "text": "surface"},
                {"type": "text_interaction", "pred": "B", "weight": 1.0, "text": "text evidence"},
                {"type": "literal_challenge", "pred": "B", "weight": 1.2, "text": "deeper meaning"},
            ],
        }

        prediction, info = calibrate_cross_view_prediction(sample, baseline)

        self.assertEqual("B", prediction)
        self.assertTrue(info["changed"])
        self.assertEqual("complementary_consensus", info["candidate_gate"])

    def test_dihedral_mapping_returns_original_option(self):
        from evidence_calibration import map_prediction_to_original

        self.assertEqual("C", map_prediction_to_original("A", [2, 0, 1]))


class DataLoaderTests(unittest.TestCase):
    def test_resolves_images_relative_to_repository(self):
        from data_loader import resolve_image_path

        sample = {
            "local_path": "Cimages/test-1.jpg",
            "meta_data": {"language": "zh"},
        }

        path = resolve_image_path(sample)

        self.assertTrue(path.is_absolute())
        self.assertEqual(ROOT / "images" / "Cimages" / "test-1.jpg", path)


class ModelAdapterTests(unittest.TestCase):
    def test_adapter_imports_before_runtime_dependencies_are_loaded(self):
        from models import git_llava

        self.assertTrue(callable(git_llava.load_model))
        self.assertTrue(callable(git_llava.infer))


class PipelineSmokeTests(unittest.TestCase):
    def test_one_sample_runs_through_all_three_stages(self):
        import evidence_calibration
        import eval_mercury
        import eval_mercury_sub
        import infer_mercury

        original_load_model = infer_mercury.load_model
        original_infer = infer_mercury.infer
        infer_mercury.load_model = lambda *_args, **_kwargs: {"ready": True}
        infer_mercury.infer = lambda _name: (
            lambda _prompts, **_kwargs: ["A", "B", "B", "A"]
        )
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                result_root = Path(temporary_directory)
                multi_view_dir = result_root / "multi_view"
                cvec_dir = result_root / "cvec"
                pesc_dir = result_root / "pesc"
                common = {
                    "model_name": "git-llava-13b",
                    "config": str(ROOT / "src" / "config" / "config_mercury.yaml"),
                    "split": ["C_metaphors"],
                    "mode": ["none"],
                    "batch_size": 1,
                    "num_workers": 1,
                    "infer_limit": 1,
                    "use_accel": False,
                }
                infer_mercury.run(
                    SimpleNamespace(
                        **common,
                        model_path=None,
                        output_dir=str(multi_view_dir),
                    )
                )
                evidence_calibration.run_stage(
                    SimpleNamespace(
                        **common,
                        stage="cvec",
                        output_dir=str(cvec_dir),
                        baseline_result_dir=str(multi_view_dir),
                        baseline_model_name="git-llava-13b",
                        answer_max_new_tokens=32,
                    )
                )
                evidence_calibration.run_stage(
                    SimpleNamespace(
                        **common,
                        stage="pesc",
                        output_dir=str(pesc_dir),
                        baseline_result_dir=str(multi_view_dir),
                        baseline_model_name="git-llava-13b",
                        answer_max_new_tokens=32,
                    )
                )

                for path in (
                    multi_view_dir / "git-llava-13b_C_metaphors_none.jsonl",
                    cvec_dir / "git-llava-13b-cvec_C_metaphors_none.jsonl",
                    pesc_dir / "git-llava-13b-pesc_C_metaphors_none.jsonl",
                ):
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)

                evaluated_dir = result_root / "evaluated"
                eval_mercury.run(pesc_dir, evaluated_dir)
                subgroup_path = evaluated_dir / "mercury_results_table.txt"
                eval_mercury_sub.run(evaluated_dir, subgroup_path)
                self.assertTrue((evaluated_dir / "evaluation_summary.txt").is_file())
                self.assertTrue(subgroup_path.is_file())
        finally:
            infer_mercury.load_model = original_load_model
            infer_mercury.infer = original_infer


if __name__ == "__main__":
    unittest.main()
