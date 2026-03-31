import os
import argparse
import itertools
import numpy as np
import pandas as pd

from furnace_model import FurnaceModel, NormalizedFurnaceModel
from parameters import create_standard_case, quick_modify

RESULT_VARS = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']
SUMMARY_KEYS = ['T_out', 't_out', 'fs_out', 'fl_out', 'x_out', 'y_out', 'w_out', 'rhob_out', 'p_bottom']


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return value


def parse_value_list(spec):
    values = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        values.append(safe_float(item))
    return values


def read_reference_profile(reference_path, z_col='z', variables=None):
    if reference_path is None:
        return None
    df = pd.read_csv(reference_path)
    if variables is None:
        variables = RESULT_VARS
    missing = set([z_col] + variables) - set(df.columns)
    if missing:
        raise ValueError(f"参考结果文件缺少列: {sorted(missing)}")
    return df[[z_col] + variables].copy()


def model_output_path(params):
    filename = f"{params.case_name}_{params.H0:.1f}-{params.HH:.1f}m.csv"
    return os.path.join(os.getcwd(), filename)


def compare_profiles(df_ref, df_model, z_col='z', variables=None):
    if variables is None:
        variables = RESULT_VARS

    z_ref = df_ref[z_col].values
    z_model = df_model[z_col].values
    if len(z_model) < 2:
        raise ValueError('模型输出必须至少包含两个 z 点以进行插值比较。')

    errors = {}
    all_rmse = []
    for var in variables:
        y_ref = df_ref[var].values
        y_model = np.interp(z_ref, z_model, df_model[var].values)
        diff = y_model - y_ref
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        max_err = np.max(np.abs(diff))
        errors[var] = {'rmse': float(rmse), 'mae': float(mae), 'max': float(max_err)}
        all_rmse.append(rmse)

    total_rmse = float(np.sqrt(np.mean(np.array(all_rmse) ** 2))) if all_rmse else 0.0
    return errors, total_rmse


def compare_summary(result, reference_summary):
    common = [k for k in SUMMARY_KEYS if k in reference_summary and k in result]
    if not common:
        raise ValueError('参考汇总结果与模型结果没有可比较的字段。')
    diffs = []
    details = {}
    for key in common:
        r = float(result[key])
        t = float(reference_summary[key])
        diff = r - t
        diffs.append(diff ** 2)
        details[key] = {'result': r, 'reference': t, 'error': float(abs(diff))}
    rmse = float(np.sqrt(np.mean(diffs)))
    return details, rmse


def load_model_result_csv(params):
    path = model_output_path(params)
    if not os.path.exists(path):
        raise FileNotFoundError(f'模型结果文件不存在: {path}')
    return pd.read_csv(path)


def run_case(params, model_class=FurnaceModel, save_parameters=False):
    model = model_class(params)
    results = model.run()
    if save_parameters:
        from save_load import save_parameters as _save
        _save(params)
    return results


def summary_from_reference_csv(reference_path):
    df = pd.read_csv(reference_path)
    if {'T_out', 't_out', 'fs_out', 'p_bottom'}.issubset(df.columns):
        return df.iloc[0].to_dict()
    return None


def build_param_name(base_name, param_name, value):
    if isinstance(value, float):
        return f"{base_name}_{param_name}_{value:.6g}"
    return f"{base_name}_{param_name}_{value}"


def run_parameter_sensitivity(base_params,
                              param_name,
                              values,
                              model_class=FurnaceModel,
                              reference_path=None,
                              output_csv=None):
    df_ref = read_reference_profile(reference_path) if reference_path else None
    reference_summary = None
    if reference_path:
        try:
            reference_summary = summary_from_reference_csv(reference_path)
        except Exception:
            reference_summary = None

    records = []
    for value in values:
        name = build_param_name(base_params.case_name, param_name, value)
        params = quick_modify(base_params, case_name=name, **{param_name: value})
        print(f"运行参数: {param_name}={value}, case_name={params.case_name}")
        result = run_case(params, model_class=model_class)

        record = {'param': param_name, 'value': value, 'case_name': params.case_name}
        record.update(result)

        if df_ref is not None:
            df_model = load_model_result_csv(params)
            _, total_rmse = compare_profiles(df_ref, df_model)
            record['profile_rmse'] = total_rmse
        elif reference_summary is not None:
            _, total_rmse = compare_summary(result, reference_summary)
            record['summary_rmse'] = total_rmse

        records.append(record)

    df = pd.DataFrame(records)
    if output_csv is None:
        output_csv = f'cases/sensitivity_{param_name}.csv'
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f'敏感度分析结果已保存: {output_csv}')
    return df


def run_parameter_grid_search(base_params,
                              param_grid,
                              model_class=FurnaceModel,
                              reference_path=None,
                              output_csv=None,
                              max_cases=None):
    df_ref = read_reference_profile(reference_path) if reference_path else None
    reference_summary = None
    if reference_path:
        try:
            reference_summary = summary_from_reference_csv(reference_path)
        except Exception:
            reference_summary = None

    keys = list(param_grid.keys())
    values_list = [param_grid[k] for k in keys]
    records = []
    for combo in itertools.product(*values_list):
        if max_cases is not None and len(records) >= max_cases:
            break
        changes = dict(zip(keys, combo))
        name_parts = [f"{k}_{changes[k]:.6g}" if isinstance(changes[k], float) else f"{k}_{changes[k]}" for k in keys]
        case_name = base_params.case_name + '_' + '_'.join(name_parts)
        params = quick_modify(base_params, case_name=case_name, **changes)
        print(f"运行组合: {changes}")
        result = run_case(params, model_class=model_class)

        record = {'case_name': params.case_name}
        record.update(changes)
        record.update(result)

        if df_ref is not None:
            df_model = load_model_result_csv(params)
            _, total_rmse = compare_profiles(df_ref, df_model)
            record['profile_rmse'] = total_rmse
        elif reference_summary is not None:
            _, total_rmse = compare_summary(result, reference_summary)
            record['summary_rmse'] = total_rmse

        records.append(record)

    df = pd.DataFrame(records)
    sort_key = 'profile_rmse' if 'profile_rmse' in df.columns else 'summary_rmse'
    if sort_key in df.columns:
        df = df.sort_values(sort_key)

    if output_csv is None:
        output_csv = 'cases/grid_search_results.csv'
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f'网格搜索结果已保存: {output_csv}')
    return df


def parse_param_grid(specs):
    grid = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f"参数网格规范错误: {spec}")
        key, values = spec.split('=', 1)
        grid[key.strip()] = parse_value_list(values)
    return grid


def parse_grid_range(spec):
    parts = spec.split(':')
    if len(parts) != 3:
        raise ValueError('网格范围格式应为 start:end:step，例如 0:20:0.01')
    start, end, step = [float(x) for x in parts]
    if step <= 0:
        raise ValueError('步长必须大于 0')
    return np.arange(start, end + step * 0.9999999, step)


def convert_reference_xlsx(input_path, output_csv='reference.csv', z_col='z', grid=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'输入文件不存在: {input_path}')

    xl = pd.ExcelFile(input_path)
    data = {}
    z_values = []

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        if df.shape[1] < 2:
            continue

        z = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        var_name = str(df.columns[1]).strip()

        mask = ~np.isnan(z) & ~np.isnan(y)
        if not np.any(mask):
            continue

        z = z[mask]
        y = y[mask]
        if len(z) == 0:
            continue

        idx = np.argsort(z)
        z = z[idx]
        y = y[idx]

        data[var_name] = (z, y)
        z_values.append(z)

    if not data:
        raise ValueError('未从 Excel 文件中提取到有效数据。')

    if grid is None:
        z_grid = np.unique(np.concatenate(z_values))
    else:
        z_grid = np.asarray(grid, dtype=float)

    z_grid = np.sort(z_grid)
    output_df = pd.DataFrame({z_col: z_grid})

    for var_name, (z, y) in data.items():
        if len(z) == 1:
            y_interp = np.full_like(z_grid, y[0], dtype=float)
        else:
            y_interp = np.interp(z_grid, z, y, left=y[0], right=y[-1])
        output_df[var_name] = y_interp

    for var_name in RESULT_VARS:
        if var_name not in output_df.columns:
            output_df[var_name] = np.nan

    if output_csv:
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f'已保存参考 CSV: {output_csv}')

    return output_df


def main():
    parser = argparse.ArgumentParser(description='高炉参数敏感度与校准工具')
    subparsers = parser.add_subparsers(dest='command')

    parser_sens = subparsers.add_parser('sensitivity', help='单参数敏感度分析')
    parser_sens.add_argument('--param', required=True, help='要分析的参数名，例如 U, epsilon')
    parser_sens.add_argument('--values', required=True, help='参数值列表，用逗号分隔，例如 8,9,10,11,12')
    parser_sens.add_argument('--reference', help='参考结果CSV文件路径')
    parser_sens.add_argument('--model', choices=['physical', 'normalized'], default='physical', help='使用模型类型')
    parser_sens.add_argument('--output', help='敏感度结果输出CSV路径')

    parser_grid = subparsers.add_parser('grid', help='参数组合网格搜索')
    parser_grid.add_argument('--params', required=True, nargs='+', help='参数网格规格，例如 U=8,9,10 epsilon=0.20,0.22')
    parser_grid.add_argument('--reference', help='参考结果CSV文件路径')
    parser_grid.add_argument('--model', choices=['physical', 'normalized'], default='physical', help='使用模型类型')
    parser_grid.add_argument('--output', help='网格搜索结果输出CSV路径')
    parser_grid.add_argument('--max_cases', type=int, help='最大求解数量')

    parser_convert = subparsers.add_parser('convert', help='将 Muchi1970b_xO2 Excel 转换成参考 CSV')
    parser_convert.add_argument('--input', required=True, help='输入 Excel 文件路径')
    parser_convert.add_argument('--output', default='reference.csv', help='输出 CSV 文件路径')
    parser_convert.add_argument('--grid', help='可选统一 z 网格，格式为 start:end:step，例如 0:20:0.01')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command == 'convert':
        grid = parse_grid_range(args.grid) if args.grid else None
        df = convert_reference_xlsx(args.input, output_csv=args.output, grid=grid)
        print(df.head())
        return

    base_params = create_standard_case('default')
    model_class = NormalizedFurnaceModel if getattr(args, 'model', None) == 'normalized' else FurnaceModel

    if args.command == 'sensitivity':
        values = parse_value_list(args.values)
        df = run_parameter_sensitivity(base_params,
                                       args.param,
                                       values,
                                       model_class=model_class,
                                       reference_path=args.reference,
                                       output_csv=args.output)
        print(df)
    elif args.command == 'grid':
        param_grid = parse_param_grid(args.params)
        df = run_parameter_grid_search(base_params,
                                      param_grid,
                                      model_class=model_class,
                                      reference_path=args.reference,
                                      output_csv=args.output,
                                      max_cases=args.max_cases)
        print(df.head(20))


if __name__ == '__main__':
    main()
