import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

from scipy.integrate import solve_ivp,solve_bvp
import matplotlib.pyplot as plt

def reduce_ode_system(fun_9var, bc_9var, known_vars, known_solutions, z_known, var_names=None):
    """
    将9变量ODE系统降为7变量系统
    
    Args:
        fun_9var: 原始9变量函数
        bc_9var: 原始9变量边界条件
        known_vars: 已知变量的索引列表，如 [0, 1]
        known_solutions: 已知变量的解值列表 [T_values, t_values]
        z_known: 已知变量的位置列表
        var_names: 变量名称列表（可选）

    Returns:
        fun_7var(z, y): 7变量微分方程
        bc_7var(ya, yb): 7变量边界条件
    """
    
    # 创建已知变量的插值函数
    T_values, t_values = known_solutions
    # T_sol = interp1d(z_known, T_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
    # t_sol = interp1d(z_known, t_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
    T_sol = CubicSpline(z_known, T_values)
    t_sol = CubicSpline(z_known, t_values)
    known_solutions = [T_sol, t_sol]
    # print(type(known_solutions))
    # print(type(known_solutions[0]))
    # print(type(known_solutions[1](0)))
    # print(known_solutions[1](0))
    
    def fun_7var(z, y):
        """
        7变量微分方程
        
        Args:
            z: 位置（可以与z_known不同） (n,)
            y: 待求变量值               (7,n)
        
        Returns:
            remove_known_derivatives(dydx_full, known_vars): 导数值
        """
        # 重建完整的y向量（用于原始函数）
        y_full = reconstruct_full_y(z, y, known_vars, known_solutions)
        
        # 计算完整的导数
        dydx_full = fun_9var(z, y_full)
        
        # 移除已知变量对应的导数
        return remove_known_derivatives(dydx_full, known_vars)
    
    def bc_7var(ya, yb):
        """
        7变量边界条件
        
        Args:
            ya: 7变量边界值列表（左端点）
            yb: 7变量边界值列表（右端点）
        
        Returns:
            7变量边界条件
        """
        # return np.array([ya[0]-params.fs0,
        #              ya[1]-params.fl0,
        #              yb[2]-params.xH,
        #              yb[3]-params.yH,
        #              yb[4]-params.wH,
        #              ya[5]-params.rho_b0,
        #              yb[6]-params.pH])
        H1 = z_known[0]
        H2 = z_known[-1]
        # 计算完整的边界条件
        ya_full = reconstruct_full_y(H1, ya, known_vars, known_solutions)
        yb_full = reconstruct_full_y(H2, yb, known_vars, known_solutions)
        
        return np.delete(bc_9var(ya_full, yb_full), known_vars, axis=0)
    
    return fun_7var, bc_7var

# 辅助函数
def reconstruct_full_y(z, y_reduced, known_vars, known_solutions):
    """重建完整的y向量"""

    if not np.isscalar(z):
        y_full = np.empty((9, len(z)))
    else:
        y_full = np.empty(9)
    
    reduced_idx = 0
    
    for i in range(9):
        if i in known_vars:
            # 使用已知解
            known_idx = known_vars.index(i)
            y_full[i] = known_solutions[known_idx](z)
        else:
            # 使用待求解变量
            y_full[i] = y_reduced[reduced_idx]
            reduced_idx += 1
    
    return y_full

def remove_known_derivatives(dydx_full, known_vars):
    """移除已知变量的导数"""
    return np.delete(dydx_full, known_vars, axis=0)


def reduce_ode_system_5(fun_9var, bc_9var, known_vars, known_solutions, z_known, var_names=None):
    """
    将9变量ODE系统降为5变量系统
    
    Args:
        fun_9var: 原始9变量函数
        bc_9var: 原始9变量边界条件
        known_vars: 已知变量的索引列表，如 [0, 1, 4, 5]
        known_solutions: 已知变量的解值列表 [T_values, t_values, x_values, y_values]
        z_known: 已知变量的位置列表
        var_names: 变量名称列表（可选）

    Returns:
        fun_5var(z, y): 5变量微分方程
        bc_5var(ya, yb): 5变量边界条件
    """
    
    # 创建已知变量的插值函数
    T_values, t_values, x_values, y_values = known_solutions
    T_sol = CubicSpline(z_known, T_values)
    t_sol = CubicSpline(z_known, t_values)
    x_sol = CubicSpline(z_known, x_values)
    y_sol = CubicSpline(z_known, y_values)
    known_solutions = [T_sol, t_sol, x_sol, y_sol]
    
    def fun_5var(z, y):
        """
        5变量微分方程
        
        Args:
            z: 位置（可以与z_known不同） (n,)
            y: 待求变量值               (5,n)
        
        Returns:
            remove_known_derivatives(dydx_full, known_vars): 导数值
        """
        # 重建完整的y向量（用于原始函数）
        y_full = reconstruct_full_y(z, y, known_vars, known_solutions)
        
        # 计算完整的导数
        dydx_full = fun_9var(z, y_full)
        
        # 移除已知变量对应的导数
        return remove_known_derivatives(dydx_full, known_vars)
    
    def bc_5var(ya, yb):
        """
        5变量边界条件
        
        Args:
            ya: 5变量边界值列表（左端点）
            yb: 5变量边界值列表（右端点）
        
        Returns:
            5变量边界条件
        """
        # return np.array([ya[0]-params.fs0,
        #              ya[1]-params.fl0,
        #              yb[2]-params.xH,
        #              yb[3]-params.yH,
        #              yb[4]-params.wH,
        #              ya[5]-params.rho_b0,
        #              yb[6]-params.pH])
        H1 = z_known[0]
        H2 = z_known[-1]
        # 计算完整的边界条件
        ya_full = reconstruct_full_y(H1, ya, known_vars, known_solutions)
        yb_full = reconstruct_full_y(H2, yb, known_vars, known_solutions)
        
        return np.delete(bc_9var(ya_full, yb_full), known_vars, axis=0)
    
    return fun_5var, bc_5var

def reduce_ode_system_8(fun_9var, bc_9var, known_vars, known_solutions, z_known, var_names=None):
    """
    将9变量ODE系统降为8变量系统
    
    Args:
        fun_9var: 原始9变量函数
        bc_9var: 原始9变量边界条件
        known_vars: 已知变量的索引列表，如 [6]
        known_solutions: 已知变量的解值列表 [w_values]
        z_known: 已知变量的位置列表
        var_names: 变量名称列表（可选）

    Returns:
        fun_8var(z, y): 8变量微分方程
        bc_8var(ya, yb): 8变量边界条件
    """
    
    # 创建已知变量的插值函数
    w_values = known_solutions
    w_sol = CubicSpline(z_known, w_values)
    known_solutions = [w_sol]
    
    def fun_8var(z, y):
        """
        8变量微分方程
        
        Args:
            z: 位置（可以与z_known不同） (n,)
            y: 待求变量值               (8,n)
        
        Returns:
            remove_known_derivatives(dydx_full, known_vars): 导数值
        """
        # 重建完整的y向量（用于原始函数）
        y_full = reconstruct_full_y(z, y, known_vars, known_solutions)
        
        # 计算完整的导数
        dydx_full = fun_9var(z, y_full)
        
        # 移除已知变量对应的导数
        return remove_known_derivatives(dydx_full, known_vars)
    
    def bc_8var(ya, yb):
        """
        8变量边界条件
        
        Args:
            ya: 8变量边界值列表（左端点）
            yb: 8变量边界值列表（右端点）
        
        Returns:
            8变量边界条件
        """
        # return np.array([ya[0]-params.fs0,
        #              ya[1]-params.fl0,
        #              yb[2]-params.xH,
        #              yb[3]-params.yH,
        #              yb[4]-params.wH,
        #              ya[5]-params.rho_b0,
        #              yb[6]-params.pH])
        H1 = z_known[0]
        H2 = z_known[-1]
        # 计算完整的边界条件
        ya_full = reconstruct_full_y(H1, ya, known_vars, known_solutions)
        yb_full = reconstruct_full_y(H2, yb, known_vars, known_solutions)
        
        return np.delete(bc_9var(ya_full, yb_full), known_vars, axis=0)
    
    return fun_8var, bc_8var

def reduce_ode_system_1(fun_9var, bc_9var, known_vars, known_solutions, z_known, var_names=None):
    """
    将9变量ODE系统降为1变量系统
    
    Args:
        fun_9var: 原始9变量函数
        bc_9var: 原始9变量边界条件
        known_vars: 已知变量的索引列表，如 [0,1,2,3,4,5,6,7]
        known_solutions: 已知变量的解值列表 [T_values, t_values, fs_values, fl_values, x_values, y_values, w_values, rhob_values]
        z_known: 已知变量的位置列表
        var_names: 变量名称列表（可选）

    Returns:
        fun_1var(z, y): 1变量微分方程
        bc_1var(ya, yb): 1变量边界条件
    """
    
    # 创建已知变量的插值函数
    T_values, t_values, fs_values, fl_values, x_values, y_values, w_values, rhob_values = known_solutions
    T_sol = CubicSpline(z_known, T_values)
    t_sol = CubicSpline(z_known, t_values)
    fs_sol = CubicSpline(z_known, fs_values)
    fl_sol = CubicSpline(z_known, fl_values)
    x_sol = CubicSpline(z_known, x_values)
    y_sol = CubicSpline(z_known, y_values)
    w_sol = CubicSpline(z_known, w_values)
    rhob_sol = CubicSpline(z_known, rhob_values)
    known_solutions = [T_sol, t_sol, fs_sol, fl_sol, x_sol, y_sol, w_sol, rhob_sol]
    
    def fun_1var(z, y):
        """
        1变量微分方程
        
        Args:
            z: 位置（可以与z_known不同） (n,)
            y: 待求变量值               (1,n)
        
        Returns:
            remove_known_derivatives(dydx_full, known_vars): 导数值
        """
        # 重建完整的y向量（用于原始函数）
        y_full = reconstruct_full_y(z, y, known_vars, known_solutions)
        
        # 计算完整的导数
        dydx_full = fun_9var(z, y_full)
        
        # 移除已知变量对应的导数
        return remove_known_derivatives(dydx_full, known_vars)
    
    def bc_1var(ya, yb):
        """
        1变量边界条件
        
        Args:
            ya: 1变量边界值列表（左端点）
            yb: 1变量边界值列表（右端点）
        
        Returns:
            1变量边界条件
        """
        # return np.array([ya[0]-params.fs0,
        #              ya[1]-params.fl0,
        #              yb[2]-params.xH,
        #              yb[3]-params.yH,
        #              yb[4]-params.wH,
        #              ya[5]-params.rho_b0,
        #              yb[6]-params.pH])
        H1 = z_known[0]
        H2 = z_known[-1]
        # 计算完整的边界条件
        ya_full = reconstruct_full_y(H1, ya, known_vars, known_solutions)
        yb_full = reconstruct_full_y(H2, yb, known_vars, known_solutions)
        
        return np.delete(bc_9var(ya_full, yb_full), known_vars, axis=0)
    
    return fun_1var, bc_1var