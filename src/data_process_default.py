import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
from paths import data_path, cases_path, ensure_dirs, output_path

# 分段线性分布作为初值
def multi_value_interpolation(x_control, y_control, num_output_points=2000):
    """
    使用numpy.interp处理多个控制点
    """
    # 生成输出的x坐标
    x_output = np.linspace(x_control[0], x_control[-1], num_output_points)
    
    # 线性插值得到y值
    y_output = np.interp(x_output, x_control, y_control)
    
    return x_output, y_output

from main import load_parameters
from furnace_model import FurnaceModel
params = load_parameters("my_design")   # 调用已保存的参数（由 save_load 统一管理路径）
model = FurnaceModel(params)

# 1. 初值设置（分段线性）
H0 = model.params.H0
H1 = model.params.H1
H2 = model.params.H2
H3 = model.params.H3
HH = model.params.HH

# 0 m
y0 = model.params.value0
# 4 m
y1 = model.params.value1
# 12 m
y2 = model.params.value2
# 16 m
y3 = model.params.value3
# 20 m
yH = model.params.valueH

# 问题设置
H_ctrl = [H0, H1, H2, H3, HH]

# 初始猜测（可以比较粗糙）
T_ctrl = [y0[0], y1[0], y2[0], y3[0], yH[0]]
t_ctrl = [y0[1], y1[1], y2[1], y3[1], yH[1]]
fs_ctrl = [y0[2], y1[2], y2[2], y3[2], yH[2]]
fl_ctrl = [y0[3], y1[3], y2[3], y3[3], yH[3]]
x_ctrl = [y0[4], y1[4], y2[4], y3[4], yH[4]]
y_ctrl = [y0[5], y1[5], y2[5], y3[5], yH[5]]
w_ctrl = [y0[6], y1[6], y2[6], y3[6], yH[6]]
rho_b_ctrl = [y0[7], y1[7], y2[7], y3[7], yH[7]]
p_ctrl = [y0[8], y1[8], y2[8], y3[8], yH[8]]

T = model.multi_value_interpolation(H_ctrl, T_ctrl, model.params.initial_mesh)
t = model.multi_value_interpolation(H_ctrl, t_ctrl, model.params.initial_mesh)
fs = model.multi_value_interpolation(H_ctrl, fs_ctrl, model.params.initial_mesh)
fl = model.multi_value_interpolation(H_ctrl, fl_ctrl, model.params.initial_mesh)
x = model.multi_value_interpolation(H_ctrl, x_ctrl, model.params.initial_mesh)
y = model.multi_value_interpolation(H_ctrl, y_ctrl, model.params.initial_mesh)
w = model.multi_value_interpolation(H_ctrl, w_ctrl, model.params.initial_mesh)
rhob = model.multi_value_interpolation(H_ctrl, rho_b_ctrl, model.params.initial_mesh)
p = model.multi_value_interpolation(H_ctrl, p_ctrl, model.params.initial_mesh)

z_guess = np.linspace(H0, HH, model.params.initial_mesh)

# 读取CSV文件（统一从 data/）
df = pd.read_csv(data_path("default_case_U_10_0.0-20.0m.csv"))
df_hc = pd.read_csv(data_path("test_hc_5n4_1e-3_linear_N=100.csv"))
df_hc2 = pd.read_csv(data_path("NEW_test_hc_5n4_R2_1200_linear_N=2000.csv"))
df_hc3 = pd.read_csv(data_path("test_hc_5n4_R2_1200_linear_N=500.csv"))
df_hc4 = pd.read_csv(data_path("test_hc_5n4_R2_1200_linear_N=2000.csv"))


# 生成均匀分布的索引
indices= np.linspace(0, len(df)-1, len(df), dtype=int)
indices_hc = np.linspace(0, len(df_hc)-1, len(df_hc), dtype=int)
indices_hc2 = np.linspace(0, len(df_hc2)-1, len(df_hc2), dtype=int)
indices_hc3 = np.linspace(0, len(df_hc3)-1, len(df_hc3), dtype=int)
indices_hc4 = np.linspace(0, len(df_hc4)-1, len(df_hc4), dtype=int)


# 按索引取行
sampled_df = df.iloc[indices]
sampled_df_hc = df_hc.iloc[indices_hc]
sampled_df_hc2 = df_hc2.iloc[indices_hc2]
sampled_df_hc3 = df_hc3.iloc[indices_hc3]
sampled_df_hc4 = df_hc4.iloc[indices_hc4]
sampled_init = pd.DataFrame(np.vstack((z_guess, T, t, fs, fl, x, y, w, rhob, p)).T, columns=['z', 'T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p'])

# # 保存结果
# sampled_df.to_csv('sampled_file.csv', index=False)


# # 通过工作表名称读取（统一从 data/）
xlsx_path = data_path("Muchi1970b_xO2=0.xlsx")
df_t = pd.read_excel(xlsx_path, sheet_name="Sheet1")
df_T = pd.read_excel(xlsx_path, sheet_name="Sheet2")
df_fs = pd.read_excel(xlsx_path, sheet_name="Sheet3")
df_fl = pd.read_excel(xlsx_path, sheet_name="Sheet4")
df_x = pd.read_excel(xlsx_path, sheet_name="Sheet5")
df_y = pd.read_excel(xlsx_path, sheet_name="Sheet6")
df_w = pd.read_excel(xlsx_path, sheet_name="Sheet7")
df_rhob = pd.read_excel(xlsx_path, sheet_name="Sheet8")
df_p = pd.read_excel(xlsx_path, sheet_name="Sheet9")
# 
# # 
zt, t_ref = multi_value_interpolation(np.asarray(df_t['z']), np.asarray(df_t['t']), num_output_points=len(indices))
zT, T_ref = multi_value_interpolation(np.asarray(df_T['z']), np.asarray(df_T['T']), num_output_points=len(indices))
zfs, fs_ref = multi_value_interpolation(np.asarray(df_fs['z']), np.asarray(df_fs['fs']), num_output_points=len(indices))
zfl, fl_ref = multi_value_interpolation(np.asarray(df_fl['z']), np.asarray(df_fl['fl']), num_output_points=len(indices))
zx, x_ref = multi_value_interpolation(np.asarray(df_x['z']), np.asarray(df_x['x']), num_output_points=len(indices))
zy, y_ref = multi_value_interpolation(np.asarray(df_y['z']), np.asarray(df_y['y']), num_output_points=len(indices))
zw, w_ref = multi_value_interpolation(np.asarray(df_w['z']), np.asarray(df_w['w']), num_output_points=len(indices))
zrhob, rhob_ref = multi_value_interpolation(np.asarray(df_rhob['z']), np.asarray(df_rhob['rhob']), num_output_points=len(indices))
zp, p_ref = multi_value_interpolation(np.asarray(df_p['z']), np.asarray(df_p['p']*1e4), num_output_points=len(indices))
# F_ref = multi_value_interpolation(np.asarray(df_F['z']), np.asarray(df_F['F']), num_output_points=len(indices))

z_ref = [zT, zt, zfs, zfl, zx, zy, zw, zrhob, zp]
df_ref = pd.DataFrame({'T': T_ref, 't': t_ref, 'fs': fs_ref, 'fl': fl_ref, 'x': x_ref, 'y': y_ref, 'w': w_ref, 'rhob': rhob_ref, 'p': p_ref})

plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']
# 作图
y_bvp = sampled_df
y_bvp_hc = sampled_df_hc
y_bvp_hc2 = sampled_df_hc2
y_bvp_hc3 = sampled_df_hc3
y_bvp_hc4 = sampled_df_hc4
y_init = sampled_init
plt.figure(figsize=(12, 8))
variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
y_labels = ['T(K)', 't(K)', 'fs(-)', 'fl(-)', 'x(-)', 'y(-)', 'w(-)', 'rhob(kg/m^3 bed)', 'p(Kg/m2)']
for i in range(9):
    plt.subplot(3, 3, i+1)
    # plt.plot(y_init['z'], y_init[variables[i]], label = '分段线性初值', linestyle='--') # 
    plt.plot(y_bvp['z'], y_bvp[variables[i]], label = 'bvp')       # 
    # plt.plot(y_bvp_hc['z'], y_bvp_hc[variables[i]], label = 'N=100') # 
    plt.plot(y_bvp_hc2['z'], y_bvp_hc2[variables[i]], label = 'isomorphic', linewidth=2.0) # 
    # plt.plot(y_bvp_hc3['z'], y_bvp_hc3[variables[i]], label = 'R2分段 N=500', linestyle=':',linewidth=2.0) # 
    # plt.plot(y_bvp_hc4['z'], y_bvp_hc4[variables[i]], label = 'R2分段 N=2000', linestyle=':',linewidth=2.0) # 
    plt.plot(z_ref[i], df_ref[variables[i]], label = 'Muchi1970b_xO2=0')        # Muchi1970b_xO2=0.xlsx
    plt.ylabel(y_labels[i])
    plt.xlabel('z')
    plt.legend()
plt.tight_layout()
plt.show()
plt.close()


# plt.plot(y_bvp['z'], y_bvp['T'] - y_bvp_hc['T'], label = 'T_hc - T_bvp')
# plt.plot(y_bvp['z'], y_bvp['t'] - y_bvp_hc['t'], label = 't_hc - t_bvp')
# plt.xlabel('z')
# plt.ylabel('diff(K)')
# plt.legend()
# plt.show()

# # 作图
# plt.figure(figsize=(12, 8))
# y_labels = ['diff_T(K)', 'diff_t(K)', 'diff_fs(-)', 'diff_fl(-)', 'diff_x(-)', 'diff_y(-)', 'diff_w(-)', 'diff_rhob(kg/m^3 bed)', 'diff_p(Kg/m2)']
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.plot(y_bvp['z'], (y_bvp[variables[i]] - y_bvp_2[variables[i]]))
#     plt.ylabel(y_labels[i])
#     plt.xlabel('z')
# plt.tight_layout()
# plt.show()

def plot_epsilon_sensitivity_profiles(save_fig=True):
    """
    绘制 epsilon（床层空隙率）敏感度：各算例剖面与 reference_0_20.csv 的 9 变量对比，并保存图片。
    数据来自 config/cases/sensitivity_epsilon.csv 与 data/default_case_epsilon_*_0.0-20.0m.csv。
    """
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False
    from matplotlib import font_manager
    # 设置宋体用于中文
    zh_font = font_manager.FontProperties(fname=font_manager.findSystemFonts(fontpaths=None, fontext='ttf')[0], size=13)
    for f in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'simsun' in f.lower() or 'song' in f.lower():
            zh_font = font_manager.FontProperties(fname=f, size=13)
            break

    ensure_dirs()
    df_rmse = pd.read_csv(cases_path("sensitivity_epsilon.csv"))
    ref = pd.read_csv(data_path("reference_0_20.csv"))
    variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
    latex_labels = [
        r'$\it{T}$',
        r'$\it{t}$',
        r'$\it{f}_{\it{s}}$',
        r'$\it{f}_{\it{l}}$',
        r'$\it{x}$',
        r'$\it{y}$',
        r'$\it{w}$',
        r'$\rho_{\it{b}}$',
        r'$\it{p}$'
    ]
    z_ref = ref['z'].values

    # 3:2
    if save_fig:
        outdir = output_path("260130")
        os.makedirs(outdir, exist_ok=True)
        plt.figure(figsize=(18, 12))
    else:
        plt.figure(figsize=(18, 18))

    for i, (var, label) in enumerate(zip(variables, latex_labels)):
        plt.subplot(3, 3, i+1)
        plt.plot(z_ref, ref[var], label='Reference', color='black', linewidth=2)
        y_min, y_max = np.nanmin(ref[var]), np.nanmax(ref[var])
        for _, row in df_rmse.iterrows():
            eps = row["value"]
            prof = _REPO_ROOT / "data" / (
                f"{row['case_name']}_{row['H0']:.1f}-{row['HH']:.1f}m.csv"
            )
            try:
                df = pd.read_csv(prof)
                plt.plot(df["z"], df[var], label=f"ε={eps}")
                y_min = min(y_min, np.nanmin(df[var]))
                y_max = max(y_max, np.nanmax(df[var]))
            except Exception as e:
                print(f"跳过 {prof}: {e}")
        plt.xlabel('z (m)', fontproperties=zh_font)
        plt.ylabel(label, fontproperties=None)
        plt.title(label, fontsize=14, pad=10, fontproperties=zh_font)
        plt.xlim(0, 20)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        plt.legend(fontsize=10)
        plt.grid(True)
    plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.25)
    if save_fig:
        outpath1 = os.path.join(outdir, 'sensitivity_epsilon_3_2.png')
        plt.savefig(outpath1, dpi=1000, bbox_inches='tight')
        print(f"图片已保存到 {outpath1}")
        plt.close()
        # # 4:3
        # plt.figure(figsize=(16, 12))
        # for i, (var, label) in enumerate(zip(variables, latex_labels)):
        #     plt.subplot(3, 3, i+1)
        #     plt.plot(z_ref, ref[var], label='Reference', color='black', linewidth=2)
        #     y_min, y_max = np.nanmin(ref[var]), np.nanmax(ref[var])
        #     for _, row in df_rmse.iterrows():
        #         eps = row["value"]
        #         prof = _REPO_ROOT / "data" / (
        #             f"{row['case_name']}_{row['H0']:.1f}-{row['HH']:.1f}m.csv"
        #         )
        #         try:
        #             df = pd.read_csv(prof)
        #             plt.plot(df["z"], df[var], label=f"ε={eps}")
        #             y_min = min(y_min, np.nanmin(df[var]))
        #             y_max = max(y_max, np.nanmax(df[var]))
        #         except Exception as e:
        #             print(f"跳过 {prof}: {e}")
        #     plt.xlabel('z (m)', fontproperties=zh_font)
        #     plt.ylabel(label, fontproperties=None)
        #     plt.title(label, fontsize=14, pad=10, fontproperties=zh_font)
        #     plt.xlim(0, 20)
        #     y_range = y_max - y_min
        #     plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        #     plt.legend(fontsize=10)
        #     plt.grid(True)
        # plt.subplots_adjust(top=0.93, hspace=0.35, wspace=0.25)
        # outpath2 = os.path.join(outdir, 'sensitivity_epsilon_4_3.png')
        # plt.savefig(outpath2, dpi=1000, bbox_inches='tight')
        # print(f"图片已保存到 {outpath2}")
        # plt.close()
    else:
        plt.show()

# 示例调用
plot_epsilon_sensitivity_profiles()

