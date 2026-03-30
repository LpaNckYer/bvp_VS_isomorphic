import os
import glob
import pandas as pd
import numpy as np
from parameter_calibration import RESULT_VARS, compare_profiles

REFERENCE = 'reference_0_20.csv'
RESULT_PATTERN = 'default_case_U_*_0.0-20.0m.csv'
OUTPUT_CSV = 'cases/sensitivity_U_profile_rmse.csv'

# 1. 读取参考文件
ref = pd.read_csv(REFERENCE)

# 2. 查找所有结果文件
files = sorted(glob.glob(RESULT_PATTERN))

records = []
for f in files:
    try:
        df = pd.read_csv(f)
        _, rmse = compare_profiles(ref, df)
        # 提取 U 值
        u_str = f.split('_U_')[1].split('_')[0]
        U = float(u_str)
        records.append({'U': U, 'file': f, 'profile_rmse': rmse})
        print(f"{f}: U={U}, profile_rmse={rmse:.6g}")
    except Exception as e:
        print(f"跳过 {f}: {e}")

if records:
    df_out = pd.DataFrame(records).sort_values('U')
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"已保存: {OUTPUT_CSV}")
else:
    print("未找到有效结果文件。")
