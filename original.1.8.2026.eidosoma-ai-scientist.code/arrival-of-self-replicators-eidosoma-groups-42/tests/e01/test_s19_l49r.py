import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts/e01/run_s19_l49r_longitudinal_process_committor_repair.py"
    spec = importlib.util.spec_from_file_location("test_l49r_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._configure()
    return module


def test_repaired_selection_has_full_f12_availability() -> None:
    module = _module()
    selected, expanded = module.select_matrices()
    assert len(selected) == 40
    assert len(expanded) == 400
    assert selected["minimumFutureFissions"].min() >= 12


def test_repair_preserves_landmarks_and_replaces_only_unavailable_selection() -> None:
    module = _module()
    selected, expanded = module.select_matrices()
    failed = module.pd.read_parquet(module.FAILED_L49_ROOT / "matrix_selection_registry.parquet")
    assert set(expanded["landmark"]) == {64, 96, 128, 160, 192}
    assert len(set(failed["matrixIndex"]) - set(selected["matrixIndex"])) == 1
    assert len(set(selected["matrixIndex"]) - set(failed["matrixIndex"])) == 1


def test_new_seed_root_and_failed_l49_are_immutable_inputs() -> None:
    module = _module()
    assert module.SEED_ROOT != bytes.fromhex(
        "7ce4281f7613ee8b97ea89df3c60a0f924342523125a211826ff7ea1cd515c4f"
    )
    assert module.validate_immutable_prior()["failedL49Unchanged"]
