# main.py
import logging
import sys
from rizhi import setup_logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_bvp
from numpy.linalg import norm

from parameters import create_standard_case, quick_modify
from furnace_model import NormalizedFurnaceModel, HCFurnaceModel
from save_load import save_parameters, load_parameters, list_saved_cases
from batch_run import run_batch_study

from reduced_bvp import reduce_ode_system, reduce_ode_system_5, reduce_ode_system_8, reduce_ode_system_1


def run_single_case(case_name="default_case", save_case=False):
    """运行单个算例"""
    print("=== 单个算例运行 ===")

    # # 1. 标准算例
    # print("\n1. 标准算例")
    # params1 = create_standard_case(case_name)
    # model1 = NormalizedFurnaceModel(params1)
    # results1 = model1.solve_normalized()
    # if save_case:
    #     save_parameters(params1)
    # print(f"结果: 出口还原度 {results1['fs_out']:.3f}")
    
    # # 2. 自定义算例
    # print("\n2. 自定义算例")
    # base_params = create_standard_case(case_name)
    # params2 = quick_modify(base_params, 
    #                      case_name="my_design",
    #                      epsilon=0.23,
    #                      T_we = 32 + 273)
    # model2 = NormalizedFurnaceModel(params2)
    # results2 = model2.solve_normalized()
    # if save_case:
    #     save_parameters(params2)
    # print(f"结果: 出口还原度 {results2['fs_out']:.3f}")

    # 3. 初值残差测试
    print("\n3. 初值残差测试")
    params = load_parameters(case_name)
    model3 = NormalizedFurnaceModel(params)
    results3 = model3.init_test()
    print(f"结果: 出口还原度 {results3['fs_out']:.3f}")


def run_saved_case(case_name):
    """演示加载已保存的算例"""
    print("\n" + "=" * 50)
    print("已保存算例演示")
    print("=" * 50)
    
    cases = list_saved_cases()
    if not cases:
        print("没有找到已保存的算例")
        return
    
    print(f"找到 {len(cases)} 个已保存算例:")
    for i, case_name in enumerate(cases[:3], 1):  # 只显示前3个
        print(f"{i}. {case_name}")
        print(results)  # 打印结果
    try:
        params = load_parameters(case_name)
        model = NormalizedFurnaceModel(params)
        results = model.solve_normalized()
        
    except Exception as e:
        print(f"{i}. {case_name}: 加载失败 - {e}")

def demo_batch_study():
    """演示批量运行"""
    print("\n" + "=" * 50)
    print("批量参数研究")
    print("=" * 50)
    
    # # 热损失换热系数敏感性分析
    # print("进行热损失换热系数敏感性分析...")
    # df_loss = run_batch_study("heat_loss_hp")
    # print("\n热损失换热系数分析结果:")
    # print(df_loss[['case_name', 'U', 'T_out', 't_out', 'fs_out']].to_string(index=False))
    
    # # 床层孔隙率敏感性分析
    # print("\n进行床层孔隙率敏感性分析...")
    # df_void = run_batch_study("fraction_void")
    # print("\n床层孔隙率分析结果:")
    # print(df_void[['case_name', 'epsilon', 'p_bottom']].to_string(index=False))

    # 网格无关性分析
    print("\n进行网格无关性分析...")
    df_grid = run_batch_study("grid_independence")
    print("\n网格无关性分析结果:")
    print(df_grid[['case_name', 'initial_mesh', 'T_out', 't_out', 'fs_out']].to_string(index=False))


def test_hc_Tt_raw():
    """
    """
    logging.info("测试 hc_Tt_raw")
    # 1. 初值设置（读取已保存的算例）
    # 读取CSV文件
    df = pd.read_csv('R2_1200_2e-5_raw.csv')
    # 按索引取行
    z_guess = df['z'].values   # 高度（物理的）
    T = df['T'].values
    t = df['t'].values
    fs = df['fs'].values
    fl = df['fl'].values   
    x = df['x'].values
    y = df['y'].values
    w = df['w'].values
    rhob = df['rhob'].values
    p = df['p'].values

    y_guess = np.array([T, t, fs, fl, x, y, w, rhob, p])   # 物理量
    N = len(z_guess) - 1

    params = load_parameters("my_design")   # 调用已保存的参数
    model = HCFurnaceModel(params)

    T_new, t_new = model.Tt_hc(z_guess, T, t, fs, fl, x, y, w, p, rhob)

    fun_7var, bc_7var = reduce_ode_system(model.blast_furnace_bvp, model.bc, [0, 1], [T_new, t_new], z_guess)
    sol_b = solve_bvp(
        fun_7var, bc_7var,
        z_guess,
        y_guess[[2, 3, 4, 5, 6, 7, 8], :],
        tol = 1e-3,
        max_nodes=50000,
        verbose=2
    )
    y_solved = CubicSpline(sol_b.x, sol_b.y.T)
    y_plot = y_solved(z_guess).T

    # 外层循环：Tt以外的7个变量
    count = 0
    while(norm(y_plot - y_guess[2:9, :])/norm(y_guess[2:9, :]) >= 1e-3) and (count < 10000):
        count += 1
        print(count)
        print("norm(y_plot - y_guess[2:9, :])/norm(y_guess[2:9, :]) = ", norm(y_plot - y_guess[2:9, :])/norm(y_guess[2:9, :]))
        y_guess[2:9, :] = y_plot.copy()
        fs, fl, x, y, w, rhob, p = y_plot
        T_new, t_new = model.Tt_hc(z_guess, T_new, t_new, fs, fl, x, y, w, p, rhob)

        fun_7var, bc_7var = reduce_ode_system(model.blast_furnace_bvp, model.bc, [0, 1], [T_new, t_new], z_guess)
        sol_b = solve_bvp(
            fun_7var, bc_7var,
            z_guess,
            y_guess[[2, 3, 4, 5, 6, 7, 8], :],
            tol = 1e-3,
            max_nodes=50000,
            verbose=2
        )
        y_solved = CubicSpline(sol_b.x, sol_b.y.T)
        y_plot = y_solved(z_guess).T

    logging.info("计算完成 hc_Tt_raw")
    # 结果绘图
    y_plot = np.append([T_new, t_new], y_plot, axis=0)
    plt.figure(figsize=(12, 8))
    variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.plot(z_guess, y_plot[i])
        plt.ylabel(variables[i])
        plt.xlabel('z')
    plt.tight_layout()
    plt.show()

    plt.plot(df['z'].values, df['T'].values, label='T')
    plt.plot(df['z'].values, df['t'].values, label='t')
    plt.plot(z_guess, T_new, label='Tnew')
    plt.plot(z_guess, t_new, label='tnew')
    plt.legend()
    plt.show()

    # 保存结果
    df = pd.DataFrame(np.vstack((z_guess, y_plot)).T, columns=['z'] + variables)
    df.to_csv('test_Tt_hc_raw_R2_1200.csv', index=False)

def test_hc_xy_raw():
    """
    """
    logging.info("测试 hc_xy_raw")
    # 1. 初值设置（读取已保存的算例）
    # 读取CSV文件
    df = pd.read_csv('R2_1200_2e-5_raw.csv')
    # 按索引取行
    z_guess = df['z'].values   # 高度（物理的）
    T = df['T'].values
    t = df['t'].values
    fs = df['fs'].values
    fl = df['fl'].values   
    x = df['x'].values
    y = df['y'].values
    w = df['w'].values
    rhob = df['rhob'].values
    p = df['p'].values

    y_guess = np.array([T, t, fs, fl, x, y, w, rhob, p])   # 物理量
    N = len(z_guess) - 1

    params = load_parameters("my_design")   # 调用已保存的参数
    model = HCFurnaceModel(params)

    x_new, y_new = model.xy_hc(z_guess, T, t, fs, fl, x, y, w, p)

    fun_7var, bc_7var = reduce_ode_system(model.blast_furnace_bvp, model.bc, [4, 5], [x_new, y_new], z_guess)
    sol_b = solve_bvp(
        fun_7var, bc_7var,
        z_guess,
        y_guess[[0, 1, 2, 3, 6, 7, 8], :],
        tol = 1e-3,
        max_nodes=50000,
        verbose=2
    )
    y_solved = CubicSpline(sol_b.x, sol_b.y.T)
    y_plot = y_solved(z_guess).T

    # 外层循环：Tt以外的7个变量
    count = 0
    while(norm(y_plot - y_guess[[0, 1, 2, 3, 6, 7, 8], :])/norm(y_guess[[0, 1, 2, 3, 6, 7, 8], :]) >= 1e-3) and (count < 10000):
        count += 1
        print(count)
        print("norm(y_plot - y_guess)/norm(y_guess) = ", norm(y_plot - y_guess[[0, 1, 2, 3, 6, 7, 8], :])/norm(y_guess[[0, 1, 2, 3, 6, 7, 8], :]))
        y_guess[[0, 1, 2, 3, 6, 7, 8], :] = y_plot.copy()
        T, t, fs, fl, w, rhob, p = y_plot
        x_new, y_new = model.xy_hc(z_guess, T, t, fs, fl, x_new, y_new, w, p)

        fun_7var, bc_7var = reduce_ode_system(model.blast_furnace_bvp, model.bc, [4, 5], [x_new, y_new], z_guess)
        sol_b = solve_bvp(
            fun_7var, bc_7var,
            z_guess,
            y_guess[[0, 1, 2, 3, 6, 7, 8], :],
            tol = 1e-3,
            max_nodes=50000,
            verbose=2
        )
        y_solved = CubicSpline(sol_b.x, sol_b.y.T)
        y_plot = y_solved(z_guess).T

    logging.info("计算完成 hc_xy_raw")
    # 结果绘图
    y_plot = np.concatenate((y_plot[[0, 1, 2, 3], :], [x_new, y_new], y_plot[[4, 5, 6], :]), axis=0)
    plt.figure(figsize=(12, 8))
    variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.plot(z_guess, y_plot[i])
        plt.ylabel(variables[i])
        plt.xlabel('z')
    plt.tight_layout()
    plt.show()

    plt.plot(df['z'].values, df['x'].values, label='x')
    plt.plot(df['z'].values, df['y'].values, label='y')
    plt.plot(z_guess, x_new, label='xnew')
    plt.plot(z_guess, y_new, label='ynew')
    plt.legend()
    plt.show()

    # 保存结果
    df = pd.DataFrame(np.vstack((z_guess, y_plot)).T, columns=['z'] + variables)
    # df.to_csv('test_xy_hc_raw_R2_1200.csv', index=False)


# code from copilot
# 收敛容忍度
TOL_T = 1e-3
TOL_t = 1e-3
TOL_XY = 1e-3
TOL_FL = 1e-2
TOL_W = 1e-3
TOL_FS = 1e-3
TOL_P = 1e-3
TOL_RHOB = 1e-3

# 最大迭代循环
MAX_INNER_LOOP = 100
MAX_OUTER_LOOP = 10000

def rel_err(new, old):
    denom = max(norm(old), 1e-16)
    return norm(new - old) / denom

def update_Ttxyfl(model, z, state):
    T_new, t_new = model.Tt_hc(z,state)
    x_new, y_new = model.xy_hc(z,state)
    fl_new = model.fl_hc(z,state)
    new_state = state.copy()
    new_state.update({"T": T_new, "t": t_new, "x": x_new, "y": y_new, "fl": fl_new})
    return new_state

def update_wfsprhob(model, z, state):
    w_new = model.w_hc(z,state)
    fs_new = model.fs_hc(z,state)
    p_new = model.p_hc(z,state)
    rhob_new = model.rhob_hc(z,state)
    new_state = state.copy()
    new_state.update({"w": w_new, "fs": fs_new, "p": p_new, "rhob": rhob_new})
    return new_state

def converge_ttx_yfl(model, z, state):
    for i in range(MAX_INNER_LOOP):
        next_state = update_Ttxyfl(model, z, state)

        err_T = rel_err(next_state["T"], state["T"])
        err_t = rel_err(next_state["t"], state["t"])
        err_x = rel_err(next_state["x"], state["x"])
        err_y = rel_err(next_state["y"], state["y"])
        err_fl = rel_err(next_state["fl"], state["fl"])

        if i % 5 == 0:  # 每5次输出一次
            logging.info(
                f"[inner {i+1}/{MAX_INNER_LOOP}] "
                f"err_T={err_T:.3e}, err_t={err_t:.3e}, "
                f"err_x={err_x:.3e}, err_y={err_y:.3e}, err_fl={err_fl:.3e}"
            )

        if (err_T < TOL_T and err_t < TOL_t and err_x < TOL_XY
                and err_y < TOL_XY and err_fl < TOL_FL):
            logging.info(f"inner loop converged at iter {i+1}")
            return next_state

        state = next_state

    logging.warning("inner loop hit max iterations")
    return state


def converge_full(model, z, state):
    for i in range(MAX_OUTER_LOOP):
        logging.info(f"[outer {i+1}/{MAX_OUTER_LOOP}] start")
        state = converge_ttx_yfl(model, z, state)

        next_state = update_wfsprhob(model, z, state)
        err_w = rel_err(next_state["w"], state["w"])
        err_fs = rel_err(next_state["fs"], state["fs"])
        err_p = rel_err(next_state["p"], state["p"])
        err_rhob = rel_err(next_state["rhob"], state["rhob"])

        logging.info(
            f"[outer {i+1}] err_w={err_w:.3e}, err_fs={err_fs:.3e}, "
            f"err_p={err_p:.3e}, err_rhob={err_rhob:.3e}"
        )

        if (err_w < TOL_W and err_fs < TOL_FS
                and err_p < TOL_P and err_rhob < TOL_RHOB):
            logging.info(f"outer loop converged at iter {i+1}")
            return next_state

        state = next_state

    logging.warning("outer loop hit max iterations")
    return state

def test_hc_5n4():
    logging.info("测试 hc_5n4")
    params = load_parameters("my_design")
    params2 = quick_modify(params, case_name="my_design", initial_mesh=2000)
    model = HCFurnaceModel(params2)

    z_guess, state = model._build_initial_guess()

    state = converge_full(model, z_guess, state)

    # y_plot 均保持和原来兼容顺序
    variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
    y_plot = [state[var] for var in variables]

    # 画图输出
    plt.figure(figsize=(12, 8))
    for i, var in enumerate(variables, start=1):
        plt.subplot(3, 3, i)
        plt.plot(z_guess, state[var], label=var)
        plt.ylabel(var)
        plt.xlabel('z')
    plt.tight_layout()
    plt.show()

    result_df = pd.DataFrame(np.vstack([z_guess] + y_plot).T, columns=['z'] + variables)
    result_df.to_csv('NEW_test_hc_5n4_R2_1200_linear_N=2000.csv', index=False)
    logging.info("test_hc_5n4 done")

def main():
    """主函数"""
    print("高炉模拟程序")
    print("=" * 30)
    
    # 运行单个算例并保存参数
    # run_single_case(case_name="my_design", save_case=True)

    # 展示已保存的算例并运行保存的单个算例
    # run_saved_case("default_case")

    # 批量参数研究
    

    # 热量流法求解测试
    # test_hc_Tt_raw()
    # test_hc_xy_raw()

    test_hc_5n4()

    print("\n" + "=" * 50)
    print("主程序结束")


# if __name__ == "__main__":
#     main()

# 使用示例
if __name__ == "__main__":
    # 设置日志
    logger = setup_logging('run.log')
    
    logger.info("程序开始运行")
    
    try:
        main()
        
    except Exception as e:
        logger.error(f"程序出错: {e}", exc_info=True)
    
    finally:
        logger.info("程序结束")