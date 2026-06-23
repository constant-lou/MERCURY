import importlib
import os


SUPPORTED_MODEL = "git-llava-13b"
_loaded = {}


def _model_config(model_path=None):
    path = model_path or os.environ.get("MERCURY_MODEL_PATH")
    if not path:
        raise RuntimeError(
            "Model weights are not configured. Set MERCURY_MODEL_PATH or pass --model_path."
        )
    return {
        "model_path_or_name": path,
        "model_base": os.environ.get("MERCURY_MODEL_BASE") or None,
        "conv_mode": os.environ.get("MERCURY_CONV_MODE", "llava_v1"),
    }


def load_model(choice, use_accel=False, model_path=None):
    if choice != SUPPORTED_MODEL:
        raise ValueError(f"Unsupported model: {choice}")
    cache_key = (choice, model_path or os.environ.get("MERCURY_MODEL_PATH"), use_accel)
    if cache_key not in _loaded:
        module = importlib.import_module(".git_llava", __package__)
        _loaded[cache_key] = module.load_model(
            choice,
            _model_config(model_path),
            use_accel=use_accel,
        )
    return _loaded[cache_key]


def infer(choice):
    if choice != SUPPORTED_MODEL:
        raise ValueError(f"Unsupported model: {choice}")
    module = importlib.import_module(".git_llava", __package__)
    return module.infer
