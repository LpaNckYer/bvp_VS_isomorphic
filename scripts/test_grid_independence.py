"""
网格无关性测试（两种求解路径）：

1) mode=bvp（默认）
   - FurnaceModel.run()，扫描 initial_mesh
   - 方程收敛：solve_bvp 残差 + 边界残差（见 Criteria）

2) mode=hc_5n4
   - 与 src/main.py::test_hc_5n4 一致：load_parameters + HCFurnaceModel + converge_full
   - 扫描 initial_mesh；方程收敛：外循环是否在给定容差内结束（converge_full 返回 converged）

出口稳定性判据两种模式相同（与参考网格的出口量比较）。

用法示例：
  python scripts/test_grid_independence.py --mode bvp
  python scripts/test_grid_independence.py --mode hc_5n4 --meshes 2000,1000,500,300,200
  python scripts/test_grid_independence.py --mode hc_5n4 --hc-case my_design
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (SRC, ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import matplotlib

matplotlib.use("Agg")

from furnace_model import FurnaceModel, HCFurnaceModel
from main import converge_full
from parameters import FurnaceParameters, quick_modify
from paths import ensure_dirs, logs_path
from save_load import load_parameters

OUTLETS = [
    "T_out",
    "t_out",
    "fs_out",
    "fl_out",
    "x_out",
    "y_out",
    "w_out",
    "rhob_out",
    "p_bottom",
]

# 未指定 --meshes 时使用；可按需要改短以加快试验
DEFAULT_MESHES = [2000, 1500, 1000, 800, 500, 400, 300, 200, 150, 100]


@dataclass(frozen=True)
class Criteria:
    # BVP：方程收敛判据（solve_bvp）
    require_bvp_success: bool = True
    max_rms_le: float = 1e-3
    bc_l2_le: float = 1e-6

    # 出口值稳定判据（相对变化 / 绝对变化）
    rel_le: float = 1e-3
    abs_le_fractions: float = 3e-4  # fs/fl/x/y/w


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def outlets_from_state(state: dict) -> dict:
    """与 FurnaceModel._plot_and_save_results 出口索引一致。"""
    T = state["T"]
    t = state["t"]
    fs = state["fs"]
    fl = state["fl"]
    x = state["x"]
    y = state["y"]
    w = state["w"]
    rhob = state["rhob"]
    p = state["p"]
    return {
        "T_out": float(T[0]),
        "t_out": float(t[-1]),
        "fs_out": float(fs[-1]),
        "fl_out": float(fl[-1]),
        "x_out": float(x[0]),
        "y_out": float(y[0]),
        "w_out": float(w[0]),
        "rhob_out": float(rhob[-1]),
        "p_bottom": float(p[-1]),
    }


def outlet_error_metrics(row: dict, ref: dict) -> dict[str, float]:
    tiny = 1e-12
    out: dict[str, float] = {}
    abs_diffs = []
    rel_diffs = []
    for k in OUTLETS:
        v = _safe_float(row.get(k))
        r = _safe_float(ref.get(k))
        ad = abs(v - r)
        rd = ad / max(abs(r), tiny)
        out[f"abs_diff_{k}"] = ad
        out[f"rel_diff_{k}"] = rd
        abs_diffs.append(ad)
        rel_diffs.append(rd)
    out["abs_diff_outlet_max"] = float(np.nanmax(abs_diffs))
    out["rel_diff_outlet_max"] = float(np.nanmax(rel_diffs))
    return out


def is_equation_converged(row: dict, c: Criteria) -> bool:
    if row.get("solver") == "hc_5n4":
        return row.get("status") == "success" and bool(row.get("hc_outer_converged"))

    if c.require_bvp_success and not bool(row.get("bvp_success")):
        return False
    max_rms = _safe_float(row.get("bvp_max_rms_residual_final"))
    bc_l2 = _safe_float(row.get("bvp_bc_l2_residual_final"))
    if not (np.isfinite(max_rms) and max_rms <= c.max_rms_le):
        return False
    if not (np.isfinite(bc_l2) and bc_l2 <= c.bc_l2_le):
        return False
    return True


def is_solution_stable(row: dict, c: Criteria) -> bool:
    if not np.isfinite(_safe_float(row.get("rel_diff_outlet_max"))):
        return False

    frac_keys = ["fs_out", "fl_out", "x_out", "y_out", "w_out"]
    for k in frac_keys:
        ad = _safe_float(row.get(f"abs_diff_{k}"))
        if not (np.isfinite(ad) and ad <= c.abs_le_fractions):
            return False

    if _safe_float(row.get("rel_diff_outlet_max")) > c.rel_le:
        return False
    return True


def run_one_bvp(mesh: int, base: FurnaceParameters, workdir: Path, bvp_verbose: int = 0) -> dict:
    p = deepcopy(base)
    p.initial_mesh = int(mesh)
    p.case_name = f"grid_{mesh}"

    model = FurnaceModel(p)
    model.bvp_verbose = bvp_verbose

    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        t0 = time.perf_counter()
        status = "success"
        try:
            results = model.run()
        except Exception as e:
            status = f"fail: {e}"
            results = getattr(model, "results", {}) or {}
        elapsed = time.perf_counter() - t0
    finally:
        os.chdir(cwd)

    row = {
        "solver": "bvp",
        "initial_mesh": mesh,
        "status": status,
        "elapsed_s": elapsed,
        "hc_outer_converged": None,
        **{k: results.get(k) for k in (["bvp_success", "bvp_tol_final", "bvp_n_nodes_final", "bvp_max_rms_residual_final", "bvp_bc_l2_residual_final"] + OUTLETS)},
    }
    return row


def run_one_hc_5n4(mesh: int, hc_case: str) -> dict:
    """对齐 main.test_hc_5n4：saved case + quick_modify(initial_mesh) + converge_full。"""
    params = load_parameters(hc_case)
    params2 = quick_modify(params, case_name=hc_case, initial_mesh=int(mesh))
    model = HCFurnaceModel(params2)

    t0 = time.perf_counter()
    status = "success"
    hc_outer_converged = False
    outlets: dict = {k: None for k in OUTLETS}
    try:
        z_guess, state = model._build_initial_guess()
        state, hc_outer_converged = converge_full(model, z_guess, state)
        outlets = outlets_from_state(state)
    except Exception as e:
        status = f"fail: {e}"
    elapsed = time.perf_counter() - t0

    return {
        "solver": "hc_5n4",
        "hc_case": hc_case,
        "initial_mesh": mesh,
        "status": status,
        "elapsed_s": elapsed,
        "hc_outer_converged": hc_outer_converged,
        "bvp_success": None,
        "bvp_tol_final": None,
        "bvp_n_nodes_final": None,
        "bvp_max_rms_residual_final": None,
        "bvp_bc_l2_residual_final": None,
        **outlets,
    }


def parse_meshes(s: str | None) -> list[int]:
    if not s or not str(s).strip():
        return list(DEFAULT_MESHES)
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="网格无关性：BVP 或 HC(test_hc_5n4 路径)")
    parser.add_argument("--mode", choices=["bvp", "hc_5n4"], default="bvp")
    parser.add_argument(
        "--meshes",
        default=None,
        help=f"逗号分隔的 initial_mesh 列表；默认 {','.join(map(str, DEFAULT_MESHES))}",
    )
    parser.add_argument(
        "--hc-case",
        default="my_design",
        help="HC 模式下调用的 save_load.load_parameters(case_name)，需与 test_hc_5n4 一致",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="输出 CSV 路径；默认 logs/grid_independence_<mode>.csv",
    )
    parser.add_argument("--bvp-verbose", type=int, default=0, help="FurnaceModel.solve_bvp verbose")
    args = parser.parse_args()

    ensure_dirs()
    meshes = parse_meshes(args.meshes)
    log_csv = Path(args.log) if args.log else logs_path(f"grid_independence_{args.mode}.csv")
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "tmp" / f"grid_independence_runs_{args.mode}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    criteria = Criteria(
        max_rms_le=1e-3,
        bc_l2_le=1e-6,
        rel_le=1e-3,
        abs_le_fractions=3e-4,
    )

    rows: list[dict] = []
    if args.mode == "bvp":
        base = FurnaceParameters()
        base.U = 10.0
        for m in meshes:
            rows.append(run_one_bvp(m, base, tmp_dir, bvp_verbose=args.bvp_verbose))
            print(f"mesh={m:4d}  status={rows[-1]['status']}  elapsed={rows[-1]['elapsed_s']:.1f}s")
            pd.DataFrame(rows).to_csv(log_csv, index=False)
    else:
        for m in meshes:
            rows.append(run_one_hc_5n4(m, args.hc_case))
            tail = rows[-1]
            print(
                f"mesh={m:4d}  status={tail['status']}  hc_outer={tail.get('hc_outer_converged')}  "
                f"elapsed={tail['elapsed_s']:.1f}s"
            )
            pd.DataFrame(rows).to_csv(log_csv, index=False)

    ref_row = None
    for r in rows:
        if is_equation_converged(r, criteria):
            ref_row = r
            break

    if ref_row is None:
        print("没有找到满足方程收敛判据的参考网格，请放宽判据、检查算例或调整 meshes 顺序（先大后小）。")
        print(f"部分结果已写入 {log_csv}")
        return

    enriched = []
    for r in rows:
        rr = dict(r)
        rr["equation_converged"] = is_equation_converged(rr, criteria)
        rr.update(outlet_error_metrics(rr, ref_row))
        rr["solution_stable_vs_ref"] = is_solution_stable(rr, criteria)
        enriched.append(rr)

    df = pd.DataFrame(enriched)
    df.to_csv(log_csv, index=False)

    ok = df[(df["equation_converged"] == True) & (df["solution_stable_vs_ref"] == True)]
    if ok.empty:
        print("没有找到同时满足“方程收敛 + 出口稳定”的网格，请放宽稳定性阈值或提高参考 mesh。")
        print(f"结果已写入 {log_csv}")
        return

    recommended = int(ok.sort_values("initial_mesh").iloc[0]["initial_mesh"])
    ref_mesh = int(ref_row["initial_mesh"])

    print()
    print("=== 网格无关性结论 ===")
    print(f"mode = {args.mode}")
    print(f"reference mesh = {ref_mesh} (满足方程收敛判据)")
    print(f"recommended minimal initial_mesh = {recommended}")
    print(f"详细结果：{log_csv}")


if __name__ == "__main__":
    main()
