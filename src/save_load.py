# save_load.py
import json
import os
from pathlib import Path

from parameters import FurnaceParameters

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _normalize_stem(filename: str) -> str:
    s = filename.replace("\\", "/").strip()
    if s.endswith(".json"):
        s = s[:-5]
    return s


def _candidate_paths(stem: str) -> list[Path]:
    """按优先级列出可能的参数 JSON 路径。"""
    stem_slash = stem.replace("\\", "/")
    if "/" in stem_slash:
        rel = Path(stem_slash)
        return [
            _REPO_ROOT / rel.with_suffix(".json") if rel.suffix != ".json" else _REPO_ROOT / rel,
        ]
    return [
        _REPO_ROOT / "config" / "cases" / f"{stem}.json",
        _REPO_ROOT / "cases" / f"{stem}.json",
        Path.cwd() / "config" / "cases" / f"{stem}.json",
        Path.cwd() / "cases" / f"{stem}.json",
    ]


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def save_parameters(params, filename=None):
    """保存参数到 config/cases（与仓库布局一致）。"""
    if filename is None:
        filename = params.case_name

    stem = _normalize_stem(filename)
    out_dir = _REPO_ROOT / "config" / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{Path(stem).name}.json"

    data = {}
    for key, value in params.__dict__.items():
        data[key] = value

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"参数已保存: {filepath}")
    return str(filepath)


def load_parameters(filename):
    """从文件加载参数。支持算例名 my_design、路径 config/cases/my_design 等。"""
    raw = Path(filename)
    if raw.suffix.lower() == ".json" and raw.is_file():
        filepath = raw.resolve()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        params = FurnaceParameters(data["case_name"])
        for key, value in data.items():
            if hasattr(params, key):
                setattr(params, key, value)
        print(f"参数已加载: {filepath}")
        return params

    stem = _normalize_stem(filename)
    candidates = _candidate_paths(stem)
    # 去重且保持顺序
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)

    filepath = _first_existing(uniq)
    if filepath is None:
        tried = ", ".join(str(p) for p in uniq[:6])
        raise FileNotFoundError(f"参数文件不存在（已尝试）: {tried}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = FurnaceParameters(data["case_name"])
    for key, value in data.items():
        if hasattr(params, key):
            setattr(params, key, value)

    print(f"参数已加载: {filepath}")
    return params


def list_saved_cases():
    """列出所有保存的算例（config/cases 与 cases）。"""
    names: set[str] = set()
    for base in (
        _REPO_ROOT / "config" / "cases",
        _REPO_ROOT / "cases",
        Path.cwd() / "config" / "cases",
        Path.cwd() / "cases",
    ):
        if not base.is_dir():
            continue
        for file in base.iterdir():
            if file.suffix == ".json":
                names.add(file.stem)
    return sorted(names)
