"""
solve_bvp_furnace.py

使用scipy的solve_bvp（打靶法/射线法）求解furnace_model.py中定义的边值问题。
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
from scipy.integrate import solve_bvp
from furnace_model import FurnaceModel
from parameters import FurnaceParameters

# 1. 定义常微分方程组（右端函数）
def odes(z, y, model):
    # y: shape (n_var, n_z)
    # z: shape (n_z,)
    dydz = np.zeros_like(y)
    n_z = y.shape[1]
    for i in range(n_z):
        state = {
            'T': y[0, i],
            't': y[1, i],
            'fs': y[2, i],
            'fl': y[3, i],
            'x': y[4, i],
            'y': y[5, i],
            'w': y[6, i],
            'rho_b': y[7, i],
            'p': y[8, i]
        }
        dydz[0, i] = model.dTdz(z[i], state)
        dydz[1, i] = model.dtdz(z[i], state)
        dydz[2, i] = model.dfsdz(z[i], state)
        dydz[3, i] = model.dfldz(z[i], state)
        dydz[4, i] = model.dxdz(z[i], state)
        dydz[5, i] = model.dydz(z[i], state)
        dydz[6, i] = model.dwdz(z[i], state)
        dydz[7, i] = model.drhobdz(z[i], state)
        dydz[8, i] = model.dpdz(z[i], state)
    return dydz

# 2. 定义边界条件
# 例如 y[0,0]=T_top, y[0,-1]=T_bottom ...
def bc(ya, yb):
    # ya, yb: y在z=0和z=L处的取值
    # 这里以常见边界为例，实际需根据模型调整
        return np.array([yb[0]-1672,
                        ya[1]-505,
                        ya[2]-0.0,
                        ya[3]-0.0,
                        yb[4]-0.365,
                        yb[5]-0.0,
                        yb[6]-0.063,
                        ya[7]-1268,
                        ya[8]-1.433e4])

if __name__ == '__main__':
    # 3. 初始化模型和参数
    params = FurnaceParameters()
    model = FurnaceModel(params)
    
    # 4. 空间网格
    z = np.linspace(0, 20, 100)  # 0~20m, 100点
    # 5. 初始猜测
    y_init = np.zeros((9, z.size))
    y_init[0] = np.linspace(300, 1800, z.size)  # 温度线性猜测
    y_init[1] = np.linspace(300, 1800, z.size)  # 固体温度
    eps = 1e-4
    y_init[2] = np.linspace(eps, 1-eps, z.size)       # 还原度，避免0和1
    y_init[3] = np.linspace(eps, 1-eps, z.size)       # 分解度，避免0和1
    y_init[4] = 0.2                             # CO
    y_init[5] = 0.1                             # CO2
    y_init[6] = 0.05                            # H2
    y_init[7] = 1000                            # rho_b，物理允许范围
    y_init[8] = 1.5e4                           # 压力，物理允许范围

    # 6. 求解
    sol = solve_bvp(lambda z, y: odes(z, y, model), bc, z, y_init)

    if sol.status == 0:
        print('求解成功！')
        # 保存为CSV
        import pandas as pd
        var_names = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rho_b', 'p']
        df = pd.DataFrame({
            'z': sol.x
        })
        for i, name in enumerate(var_names):
            df[name] = sol.y[i]
        file_name = 'bvp_solution.csv'
        df.to_csv(file_name, index=False)
        print(f'结果已保存')
        # 可视化
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        for i, label in enumerate(var_names):
            plt.subplot(3, 3, i+1)
            plt.plot(sol.x, sol.y[i], label=label)
            plt.ylabel(var_names[i])
            plt.xlabel('z (m)')
        plt.tight_layout()
        plt.close()  # 避免显示，节省内存

    else:
        print('求解失败:', sol.message)
