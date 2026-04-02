"""
初值收敛范围测试：以参考剖面（data/default_case_U_10_0.0-20.0m.csv）在
H0,H1,H2,H3,HH 处插值得到控制点初值，再对各路状态量做比例扰动，分别测试
FurnaceModel.run() 与 HCFurnaceModel + converge_full 是否可算。
"""
from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from itertools import product
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
from parameters import FurnaceParameters
from paths import logs_path, ensure_dirs

REF_CSV = ROOT / "data" / "default_case_U_10_0.0-20.0m.csv"
OUT_CSV = ROOT / "logs" / "initial_convergence_test.csv"
TMP_RUN = ROOT / "tmp" / "initial_convergence_runs"

LOG_FILE = logs_path("initial_convergence.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
)

STATE_VARS = ["T", "t", "fs", "fl", "x", "y", "w", "rhob", "p"]

RESULTS_DIR = ROOT / "results"
OUT_CONSTANT_INLET_CSV = RESULTS_DIR / "initial_convergence_constant_inlet_FurnaceModel.csv"

# 仅跑“常数分布（9变量进口值）”测试，避免长时间扫描整批参考扰动算例。
# 若要保留原来的全量扫描，把它设为 True。
RUN_REFERENCE_SCALE_TESTS = False


def load_reference_control_points(
    csv_path: Path,
    params: FurnaceParameters,
) -> list[list[float]]:
    """在 z=H0..HH 五个控制高度上，对参考 CSV 线性插值，得到 value0..valueH（各 9 维）。"""
    df = pd.read_csv(csv_path)
    z = df["z"].to_numpy(dtype=float)
    order = np.argsort(z)
    z = z[order]
    H_ctrl = [params.H0, params.H1, params.H2, params.H3, params.HH]
    rows: list[list[float]] = []
    for zc in H_ctrl:
        row = []
        for name in STATE_VARS:
            v = df[name].to_numpy(dtype=float)[order]
            row.append(float(np.interp(zc, z, v)))
        rows.append(row)
    return rows


def apply_scale_factors(
    controls: list[list[float]], scales: tuple[float, ...]
) -> list[list[float]]:
    """
    controls: 5×9，与 value0..valueH 对应。
    scales: 9 个量各自乘在五个控制点的该分量上；fs、fl 再 clip 到 [0,1]。
    """
    out: list[list[float]] = []
    for row in controls:
        new_row = []
        for j, name in enumerate(STATE_VARS):
            x = row[j] * scales[j]
            if name in ("fs", "fl"):
                x = float(np.clip(x, 0.0, 1.0))
            new_row.append(x)
        out.append(new_row)
    return out


def params_with_controls(
    base: FurnaceParameters,
    controls: list[list[float]],
    case_name: str,
) -> FurnaceParameters:
    p = deepcopy(base)
    p.case_name = case_name
    p.value0, p.value1, p.value2, p.value3, p.valueH = deepcopy(controls)
    return p


def build_constant_inlet_controls(base: FurnaceParameters) -> list[list[float]]:
    """
    用“9个边界/进口值”构造常数分布初值：
    value0..valueH 都取同一个向量 [T_in, t_in, fs_in, fl_in, x_in, y_in, w_in, rhob_in, p_in]。
    """
    inlet = [
        base.T_in,
        base.t_in,
        base.fs_in,
        base.fl_in,
        base.x_in,
        base.y_in,
        base.w_in,
        base.rhob_in,
        base.p_in,
    ]
    return [list(inlet) for _ in range(5)]


def rmse_vs_reference(
    ref: pd.DataFrame,
    z: np.ndarray,
    state: dict[str, np.ndarray],
) -> dict[str, float]:
    zr = ref["z"].to_numpy(dtype=float)
    o = np.argsort(zr)
    zr = zr[o]
    err: dict[str, float] = {}
    for name in STATE_VARS:
        vr = ref[name].to_numpy(dtype=float)[o]
        v = np.interp(z, zr, vr)
        diff = state[name] - v
        err[f"rmse_{name}"] = float(np.sqrt(np.mean(diff**2)))
    return err


def build_test_cases(
    scales_1d: list[float],
    max_random_combos: int,
    rng: np.random.Generator,
) -> list[tuple[int, tuple[float, ...], str]]:
    """返回 [(case_id, scales_9tuple, label), ...]。"""
    n = len(STATE_VARS)
    cases: list[tuple[int, tuple[float, ...], str]] = []
    case_id = 0
    cases.append((case_id, tuple(1.0 for _ in range(n)), "baseline"))
    case_id += 1

    for k, name in enumerate(STATE_VARS):
        for s in scales_1d:
            if s == 1.0:
                continue
            tup = [1.0] * n
            tup[k] = s
            cases.append((case_id, tuple(tup), f"only_{name}_x{s}"))
            case_id += 1

    grid = [0.9, 1.0, 1.1]
    pool = [t for t in product(*([grid] * n)) if t != (1.0,) * n]
    rng.shuffle(pool)
    for tup in pool[:max_random_combos]:
        cases.append((case_id, tup, "random_combo"))
        case_id += 1

    return cases


def main():
    ensure_dirs()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    TMP_RUN.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base = FurnaceParameters()
    base.U = 10.0

    if not REF_CSV.is_file():
        raise FileNotFoundError(f"参考解不存在: {REF_CSV}")

    ref_df = pd.read_csv(REF_CSV)
    base_controls = load_reference_control_points(REF_CSV, base)

    # ===== 常数分布（9变量进口值）测试 FurnaceModel.run 是否收敛 =====
    constant_inlet_controls = build_constant_inlet_controls(base)
    constant_case_name = "ic_constant_inlet_bvp"
    params_constant = params_with_controls(base, constant_inlet_controls, constant_case_name)
    # 常数进口值初值可能较“远”，为保证测试能在可接受时间内完成，
    # 这里为该测试单独降低初始网格规模（不影响其它参考扰动算例）。
    params_constant.initial_mesh = 2000
    model_constant = FurnaceModel(params_constant)
    # 建议：把 BVP 求解器 verbose 提到 2，便于你观察 `solve_bvp` 内部迭代进度。
    model_constant.bvp_verbose = 2

    cwd = os.getcwd()
    try:
        os.chdir(TMP_RUN)
        t0_const = time.perf_counter()
        status_const = "success"
        rmse_const: dict[str, float] = {}
        try:
            model_constant.run()
        except Exception as e:
            status_const = f"fail: {e}"

        elapsed_const = time.perf_counter() - t0_const

        sol_path_const = (
            TMP_RUN / f"{constant_case_name}_{params_constant.H0:.1f}-{params_constant.HH:.1f}m.csv"
        )
        if sol_path_const.is_file() and status_const == "success":
            sol_df = pd.read_csv(sol_path_const)
            z_sol = sol_df["z"].to_numpy(dtype=float)
            state_d = {k: sol_df[k].to_numpy(dtype=float) for k in STATE_VARS}
            rmse_const = rmse_vs_reference(ref_df, z_sol, state_d)

        row_const = {
            "case_type": "constant_inlet",
            "model": "FurnaceModel",
            "status": status_const,
            "elapsed_s": elapsed_const,
            "bvp_success": model_constant.results.get("bvp_success") if hasattr(model_constant, "results") else None,
            "initial_mesh": params_constant.initial_mesh,
            "solution_csv": str(sol_path_const),
            **{f"{k}_in": getattr(base, f"{k}_in") for k in ["T", "t", "fs", "fl", "x", "y", "w", "rhob", "p"]},
            **rmse_const,
        }
        pd.DataFrame([row_const]).to_csv(OUT_CONSTANT_INLET_CSV, index=False)

        print(
            f"常数进口值测试完成：status={row_const['status']}, "
            f"bvp_success={row_const['bvp_success']} -> {OUT_CONSTANT_INLET_CSV}"
        )
        logging.info(
            "Constant inlet test finished: status=%s, bvp_success=%s, elapsed=%.2fs, csv=%s",
            row_const["status"],
            row_const["bvp_success"],
            row_const["elapsed_s"],
            OUT_CONSTANT_INLET_CSV,
        )

        if not RUN_REFERENCE_SCALE_TESTS:
            return
    finally:
        os.chdir(cwd)

    scales_1d = [0.75, 0.9, 1.0, 1.1, 1.25]
    rng = np.random.default_rng(0)
    cases = build_test_cases(scales_1d, max_random_combos=24, rng=rng)
    n_cases = len(cases)
    total_steps = 2 * n_cases
    results: list[dict] = []

    t_prog0 = time.perf_counter()
    logging.info(
        "Initial convergence test started: %d cases, ref=%s", n_cases, REF_CSV
    )
    cwd = os.getcwd()
    step_idx = 0
    try:
        os.chdir(TMP_RUN)
        for ci, (case_id, scales, label) in enumerate(cases, start=1):
            logging.info("Case %d (%s) start, scales=%s", case_id, label, scales)
            perturbed = apply_scale_factors(base_controls, scales)
            name_bvp = f"ic_bvp_{case_id}"
            name_hc = f"ic_hc_{case_id}"

            row_common = {
                "case_id": case_id,
                "label": label,
                "scales": scales,
            }

            # --- FurnaceModel (BVP) ---
            params_bvp = params_with_controls(base, perturbed, name_bvp)
            model_bvp = FurnaceModel(params_bvp)
            t0 = time.perf_counter()
            try:
                model_bvp.run()
                status_bvp = "success"
                sol_path = (
                    TMP_RUN
                    / f"{name_bvp}_{params_bvp.H0:.1f}-{params_bvp.HH:.1f}m.csv"
                )
                if sol_path.is_file():
                    sol_df = pd.read_csv(sol_path)
                    z_sol = sol_df["z"].to_numpy(dtype=float)
                    state_d = {k: sol_df[k].to_numpy(dtype=float) for k in STATE_VARS}
                    rmse_bvp = rmse_vs_reference(ref_df, z_sol, state_d)
                else:
                    rmse_bvp = {}
            except Exception as e:
                status_bvp = f"fail: {e}"
                rmse_bvp = {}
            elapsed_bvp = time.perf_counter() - t0
            step_idx += 1
            logging.info(
                "Case %d BVP: %s, elapsed=%.2fs", case_id, status_bvp, elapsed_bvp
            )

            results.append(
                {
                    **row_common,
                    "model": "FurnaceModel",
                    "status": status_bvp,
                    "elapsed_s": elapsed_bvp,
                    **{k: v for k, v in rmse_bvp.items()},
                }
            )

            # --- HCFurnaceModel ---
            params_hc = params_with_controls(base, perturbed, name_hc)
            model_hc = HCFurnaceModel(params_hc)
            z_hc, state_hc = model_hc._build_initial_guess()
            t1 = time.perf_counter()
            try:
                state_out = converge_full(model_hc, z_hc, state_hc)
                status_hc = "success"
                rmse_hc = rmse_vs_reference(ref_df, z_hc, state_out)
            except Exception as e:
                status_hc = f"fail: {e}"
                rmse_hc = {}
            elapsed_hc = time.perf_counter() - t1
            step_idx += 1
            logging.info(
                "Case %d HC: %s, elapsed=%.2fs", case_id, status_hc, elapsed_hc
            )

            case_total_s = elapsed_bvp + elapsed_hc
            results[-1]["case_total_s"] = case_total_s
            results.append(
                {
                    **row_common,
                    "model": "HCFurnaceModel",
                    "status": status_hc,
                    "elapsed_s": elapsed_hc,
                    "case_total_s": case_total_s,
                    **{k: v for k, v in rmse_hc.items()},
                }
            )

            elapsed_prog = time.perf_counter() - t_prog0
            pct = 100.0 * step_idx / total_steps
            eta_s = (
                (elapsed_prog / step_idx) * (total_steps - step_idx)
                if step_idx > 0
                else float("nan")
            )
            eta_str = (
                f"{eta_s / 60:.1f} min"
                if eta_s == eta_s and eta_s >= 60
                else (f"{eta_s:.0f} s" if eta_s == eta_s else "?")
            )
            logging.info(
                "Progress %d/%d (%.1f%%), case %d/%d id=%d (%s), "
                "BVP=%.2fs HC=%.2fs total=%.2fs ETA=%s",
                step_idx,
                total_steps,
                pct,
                ci,
                n_cases,
                case_id,
                label,
                elapsed_bvp,
                elapsed_hc,
                case_total_s,
                eta_str,
            )
            print(
                f"进度 [{step_idx}/{total_steps}] ({pct:.1f}%)  "
                f"算例 {ci}/{n_cases} id={case_id} ({label})  "
                f"BVP {elapsed_bvp:.2f}s | HC {elapsed_hc:.2f}s | "
                f"本算例合计 {case_total_s:.2f}s | ETA ~{eta_str}"
            )
            sys.stdout.flush()
    finally:
        os.chdir(cwd)

    wall_total = time.perf_counter() - t_prog0
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(
        f"完成，共 {len(results)} 条记录，总用时 {wall_total:.2f}s ({wall_total / 60:.2f} min) -> {OUT_CSV}"
    )
    logging.info(
        "Initial convergence test finished: %d records, wall=%.2fs (%.2f min), csv=%s, log=%s",
        len(results),
        wall_total,
        wall_total / 60.0,
        OUT_CSV,
        LOG_FILE,
    )


if __name__ == "__main__":
    main()
