import py_compile
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    "README.md",
    "requirements.txt",
    "run_mercury.sh",
    "src/config/config_mercury.yaml",
    "src/infer/__init__.py",
    "src/infer/infer_mercury.py",
    "src/infer/multi_view_reasoning.py",
    "src/infer/evidence_calibration.py",
    "src/infer/data_loader.py",
    "src/infer/config_wrapper.py",
    "src/eval_mercury.py",
    "src/eval_mercury_sub.py",
    "src/validate_result_coverage.py",
}

FORBIDDEN_NAME_PARTS = (
    "__pycache__",
    ".pytest_cache",
    "word" + "net",
    "how" + "net",
    "concept" + "net",
    "wi" + "ki",
    "backup",
    "infer_git_rag",
    "infer_cuegraph",
    "v10",
    "v16",
)

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".json",
}

ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/](?:Users|Windows|ProgramData)[\\/]"),
    re.compile(r"(?<![}.$A-Za-z0-9_])/(?:home|root|mnt|data)/", re.IGNORECASE),
    re.compile(r"/\d{6,}/"),
)


class RepositoryCleanlinessTests(unittest.TestCase):
    def test_required_public_files_exist(self):
        missing = sorted(path for path in REQUIRED_FILES if not (ROOT / path).is_file())
        self.assertEqual([], missing)

    def test_forbidden_files_and_directories_are_absent(self):
        violations = []
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT).as_posix().lower()
            if relative.startswith("tests/"):
                continue
            if any(part in relative for part in FORBIDDEN_NAME_PARTS):
                violations.append(relative)
        self.assertEqual([], violations)

    def test_text_files_have_no_private_paths_or_legacy_names(self):
        violations = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if "tests" in path.relative_to(ROOT).parts:
                continue
            if path.relative_to(ROOT).parts[0] in {"data", "assets"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in ABSOLUTE_PATH_PATTERNS:
                if pattern.search(text):
                    violations.append(f"{path.relative_to(ROOT)}: absolute path")
                    break
            lowered = text.lower()
            for term in (
                "vis" + "age",
                "word" + "net",
                "open" + "how" + "net",
                "concept" + "net",
                "wiki" + "pedia",
            ):
                if term in lowered:
                    violations.append(f"{path.relative_to(ROOT)}: {term}")
        self.assertEqual([], violations)

    def test_python_sources_compile(self):
        failures = []
        for path in ROOT.rglob("*.py"):
            if "tests" in path.relative_to(ROOT).parts:
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{path.relative_to(ROOT)}: {exc}")
        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
