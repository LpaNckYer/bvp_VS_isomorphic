import glob

import pandas as pd

from parameter_calibration import compare_profiles
from paths import DATA_DIR, cases_path, data_path, ensure_dirs

REFERENCE = data_path("reference_0_20.csv")
RESULT_PATTERN = str(DATA_DIR / "default_case_U_*_0.0-20.0m.csv")
OUTPUT_CSV = cases_path("sensitivity_U_profile_rmse.csv")

# 1. 读取参考文件
ref = pd.read_csv(REFERENCE)

# 2. 查找所有结果文件（data/ 下）
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
    ensure_dirs()
    df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"已保存: {OUTPUT_CSV}")
else:
    print("未找到有效结果文件。")
