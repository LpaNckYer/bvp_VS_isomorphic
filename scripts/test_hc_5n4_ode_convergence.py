"""
test_hc_5n4 的变体：外循环收敛判据改为与 BVP（FurnaceModel.run）对齐的「ODE 容差」。

判据（默认与 scripts/test_grid_independence 中 BVP 判据一致的量级）：
- max_ode_rms：9 个方程在轴向网格上，(dY/dz 数值导数 − blast_furnace_bvp 右端项) 的逐方程 RMS 的最大值 < ode_tol（默认 1e-3）
- bc_l2：物理边界条件残差的 L2 范数 < bc_tol（默认 1e-6）

内循环仍为 main.converge_ttx_yfl（与原版 HC 一致）；仅外循环结束条件由
「w/fs/p/rhob 相对变化」改为上述 ODE+BC 判据。

用法：
  python scripts/test_hc_5n4_ode_convergence.py
  python scripts/test_hc_5n4_ode_convergence.py --initial-mesh 2000 --ode-tol 1e-3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (SRC, ROOT):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from furnace_model import HCFurnaceModel
from main import MAX_OUTER_LOOP, converge_ttx_yfl, update_wfsprhob
from parameters import quick_modify
from paths import ensure_dirs, output_path
from save_load import load_parameters

STATE_ORDER = ["T", "t", "fs", "fl", "x", "y", "w", "rhob", "p"]
VARIABLES = list(STATE_ORDER)


def ode_and_bc_residual_metrics(model: HCFurnaceModel, z: np.ndarray, state: dict) -> dict:
    """
    离散意义下的 ODE 缺陷：dY/dz（np.gradient）应与 blast_furnace_bvp 一致。
    与 solve_bvp 的 RMS 残差同一量级目标时可共用阈值 1e-3。
    """
    Z = np.asarray(z, dtype=float)
    Y = np.vstack([state[k] for k in STATE_ORDER])
    f = model.blast_furnace_bvp(Z, Y)
    dYdz = np.empty_like(Y)
    for j in range(Y.shape[0]):
        dYdz[j] = np.gradient(Y[j], Z)
    r = f - dYdz
    rms_per = np.sqrt(np.mean(r * r, axis=1))
    bc_vec = model.bc(Y[:, 0], Y[:, -1])
    return {
        "max_ode_rms": float(np.max(rms_per)),
        "rms_per_equation": [float(x) for x in rms_per],
        "bc_l2": float(norm(bc_vec)),
        "bc_vec": bc_vec,
    }


def converge_full_ode_criterion(
    model: HCFurnaceModel,
    z: np.ndarray,
    state: dict,
    *,
    ode_tol: float,
    bc_tol: float,
) -> tuple[dict, bool, dict]:
    """
    外循环：inner 同 main；每轮 outer 结束后用 ODE+BC 判据决定是否收敛。
    返回 (final_state, converged, last_metrics)。
    """
    last_metrics: dict = {}
    for i in range(MAX_OUTER_LOOP):
        logging.info("[ode-outer %d/%d] start", i + 1, MAX_OUTER_LOOP)
        state = converge_ttx_yfl(model, z, state)
        next_state = update_wfsprhob(model, z, state)

        m = ode_and_bc_residual_metrics(model, z, next_state)
        last_metrics = m
        logging.info(
            "[ode-outer %d] max_ode_rms=%.3e (tol=%.1e), bc_l2=%.3e (tol=%.1e)",
            i + 1,
            m["max_ode_rms"],
            ode_tol,
            m["bc_l2"],
            bc_tol,
        )

        if m["max_ode_rms"] < ode_tol and m["bc_l2"] < bc_tol:
            logging.info("ODE+BC 判据满足，外循环收敛于 iter %d", i + 1)
            return next_state, True, m

        state = next_state

    logging.warning("ODE 外循环触顶 MAX_OUTER_LOOP，未满足 ODE/BC 判据")
    return state, False, last_metrics


def main():
    parser = argparse.ArgumentParser(description="HC 5n4 + ODE 容差收敛（对齐 BVP）")
    parser.add_argument("--hc-case", default="my_design", help="load_parameters 算例名")
    parser.add_argument("--initial-mesh", type=int, default=2000)
    parser.add_argument("--ode-tol", type=float, default=1e-3, help="max RMS(ODE 缺陷) 阈值")
    parser.add_argument("--bc-tol", type=float, default=1e-6, help="||bc||_2 阈值")
    parser.add_argument(
        "--output",
        default=None,
        help="结果 CSV 路径；默认写入 output/test_hc_5n4_ode_N=....csv",
    )
    parser.add_argument("--plot", action="store_true", help="保存 3x3 剖面图 PNG")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ensure_dirs()
    params = load_parameters(args.hc_case)
    params2 = quick_modify(
        params, case_name=args.hc_case, initial_mesh=int(args.initial_mesh)
    )
    model = HCFurnaceModel(params2)

    z_guess, state = model._build_initial_guess()
    t0 = time.perf_counter()
    state_out, converged, metrics = converge_full_ode_criterion(
        model,
        z_guess,
        state,
        ode_tol=args.ode_tol,
        bc_tol=args.bc_tol,
    )
    elapsed = time.perf_counter() - t0

    out_csv = (
        Path(args.output)
        if args.output
        else output_path(f"test_hc_5n4_ode_N={args.initial_mesh}.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    y_plot = [state_out[v] for v in VARIABLES]
    result_df = pd.DataFrame(
        np.vstack([z_guess] + y_plot).T, columns=["z"] + VARIABLES
    )
    result_df.to_csv(out_csv, index=False)

    # 最终再算一次指标写入日志
    final_m = ode_and_bc_residual_metrics(model, z_guess, state_out)
    logging.info(
        "完成: converged=%s, elapsed=%.2fs, max_ode_rms=%.3e, bc_l2=%.3e -> %s",
        converged,
        elapsed,
        final_m["max_ode_rms"],
        final_m["bc_l2"],
        out_csv,
    )

    if args.plot:
        fig_path = out_csv.with_suffix(".png")
        plt.figure(figsize=(12, 8))
        for i, var in enumerate(VARIABLES, start=1):
            plt.subplot(3, 3, i)
            plt.plot(z_guess, state_out[var], label=var)
            plt.ylabel(var)
            plt.xlabel("z")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logging.info("Saved figure %s", fig_path)

    print(
        f"converged={converged}  max_ode_rms={final_m['max_ode_rms']:.3e}  "
        f"bc_l2={final_m['bc_l2']:.3e}  csv={out_csv}"
    )


if __name__ == "__main__":
    main()
