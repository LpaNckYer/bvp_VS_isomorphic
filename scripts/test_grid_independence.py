"""
FurnaceModel.run() 网格无关性测试：
- 扫描不同 initial_mesh（初始网格点数）
- 以 solve_bvp 的残差指标（max RMS residual / BC residual）判断“方程收敛”
- 以出口值随网格变化的波动判断“解稳定性”

输出：
- logs/grid_independence.csv：逐网格记录（耗时、收敛、残差、出口值、与参考网格差异）
- 控制台打印：推荐的最小 initial_mesh（通常是“几百个点”量级）
"""

from __future__ import annotations

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

from furnace_model import FurnaceModel
from parameters import FurnaceParameters
from paths import ensure_dirs, logs_path


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


@dataclass(frozen=True)
class Criteria:
    # 方程收敛判据（结合 solve_bvp 内部容差）
    require_bvp_success: bool = True
    max_rms_le: float = 1e-3
    bc_l2_le: float = 1e-6

    # 出口值稳定判据（相对变化 / 绝对变化）
    rel_le: float = 1e-3
    abs_le_fractions: float = 3e-4  # fs/fl/x/y/w 常用更严格的 abs


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def outlet_error_metrics(row: dict, ref: dict) -> dict[str, float]:
    """
    对每个出口量，计算：
    - abs_diff
    - rel_diff（用 max(|ref|, tiny) 做分母）
    并返回整体 max。
    """
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
    # 以与 reference 的差异判断稳定性
    if not np.isfinite(_safe_float(row.get("rel_diff_outlet_max"))):
        return False

    # fractions: fs, fl, x, y, w 用 abs 约束更直观
    frac_keys = ["fs_out", "fl_out", "x_out", "y_out", "w_out"]
    for k in frac_keys:
        ad = _safe_float(row.get(f"abs_diff_{k}"))
        if not (np.isfinite(ad) and ad <= c.abs_le_fractions):
            return False

    # 其余用 max 相对误差约束
    if _safe_float(row.get("rel_diff_outlet_max")) > c.rel_le:
        return False
    return True


def run_one(mesh: int, base: FurnaceParameters, workdir: Path) -> dict:
    p = deepcopy(base)
    p.initial_mesh = int(mesh)
    p.case_name = f"grid_{mesh}"

    model = FurnaceModel(p)
    model.bvp_verbose = 0

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
        "initial_mesh": mesh,
        "status": status,
        "elapsed_s": elapsed,
        **{k: results.get(k) for k in (["bvp_success", "bvp_tol_final", "bvp_n_nodes_final", "bvp_max_rms_residual_final", "bvp_bc_l2_residual_final"] + OUTLETS)},
    }
    return row


def main():
    ensure_dirs()
    log_csv = logs_path("grid_independence.csv")
    tmp_dir = ROOT / "tmp" / "grid_independence_runs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 你可以按需增减；目标是确定“最少几百个点”。
    # 为避免低网格长时间卡住，先跑大网格作为参考，再向下扫。
    meshes = [2000, 1500, 1000, 800, 600, 500, 400, 350, 300, 250, 200, 150, 100]

    # 作为“参考真值”的网格：默认取 meshes 中最大的、且方程收敛成功的那一项
    criteria = Criteria(
        max_rms_le=1e-3,   # 对应 run() 里最终 tol=1e-3 的量级
        bc_l2_le=1e-6,
        rel_le=1e-3,
        abs_le_fractions=3e-4,
    )

    base = FurnaceParameters()
    base.U = 10.0

    rows: list[dict] = []
    for m in meshes:
        rows.append(run_one(m, base, tmp_dir))
        print(f"mesh={m:4d}  status={rows[-1]['status']}  elapsed={rows[-1]['elapsed_s']:.1f}s")
        pd.DataFrame(rows).to_csv(log_csv, index=False)

    # 选 reference（最大 mesh 且 equation_converged）
    ref_row = None
    for r in rows:
        if is_equation_converged(r, criteria):
            ref_row = r
            break

    if ref_row is None:
        print("没有找到满足方程收敛判据的参考网格，请放宽残差阈值或检查算例可收敛性。")
        return

    # 补充与 reference 的误差指标
    enriched = []
    for r in rows:
        rr = dict(r)
        rr["equation_converged"] = is_equation_converged(rr, criteria)
        rr.update(outlet_error_metrics(rr, ref_row))
        rr["solution_stable_vs_ref"] = is_solution_stable(rr, criteria)
        enriched.append(rr)

    df = pd.DataFrame(enriched)
    df.to_csv(log_csv, index=False)

    # 推荐最小 mesh：方程收敛 + 出口稳定
    ok = df[(df["equation_converged"] == True) & (df["solution_stable_vs_ref"] == True)]
    if ok.empty:
        print("没有找到同时满足“方程收敛 + 出口稳定”的网格，请放宽稳定性阈值或提高最大 mesh 作为参考。")
        print(f"结果已写入 {log_csv}")
        return

    recommended = int(ok.sort_values("initial_mesh").iloc[0]["initial_mesh"])
    ref_mesh = int(ref_row["initial_mesh"])

    print()
    print("=== 网格无关性结论 ===")
    print(f"reference mesh = {ref_mesh} (满足方程收敛判据)")
    print(f"recommended minimal initial_mesh = {recommended}")
    print(f"详细结果：{log_csv}")


if __name__ == "__main__":
    main()

