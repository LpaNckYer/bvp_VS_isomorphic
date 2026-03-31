# 批量整理项目文件的Python脚本
import os
import shutil

# 根目录
ROOT = r'd:/Group/HyOCR/20260130'

# 分类规则
move_map = {
    'src': [
        'batch_run.py', 'constant.py', 'data_process_default.py', 'furnace_model.py', 'heatcurrent_matrix_n.py',
        'heatcurrent_matrix_s.py', 'main.py', 'parameter_calibration.py', 'parameters.py', 'reaction_test.py',
        'recalc_profile_rmse.py', 'reduced_bvp.py', 'rizhi.py', 'save_load.py', 'sigmoid.py', 'simple_matrix.py'
    ],
    'tmp': [
        'tempCodeRunnerFile.py', '__pycache__'
    ],
    'data': [
        '*.csv', '*.xlsx'
    ],
    'config': [
        'cases', '*.json'
    ],
    'logs': [
        '*.log'
    ]
}

# 工具函数

def move_file(src, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    dst = os.path.join(dst_folder, os.path.basename(src))
    print(f'Move: {src} -> {dst}')
    shutil.move(src, dst)

def move_by_pattern(pattern, dst_folder):
    import glob
    for f in glob.glob(os.path.join(ROOT, pattern)):
        if os.path.isfile(f):
            move_file(f, os.path.join(ROOT, dst_folder))
        elif os.path.isdir(f):
            move_dir(f, os.path.join(ROOT, dst_folder))

def move_dir(src, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    dst = os.path.join(dst_folder, os.path.basename(src))
    print(f'Move dir: {src} -> {dst}')
    shutil.move(src, dst)

if __name__ == '__main__':
    # 精确文件
    for folder, files in move_map.items():
        for f in files:
            if '*' in f:
                move_by_pattern(f, folder)
            else:
                src = os.path.join(ROOT, f)
                if os.path.exists(src):
                    if os.path.isfile(src):
                        move_file(src, os.path.join(ROOT, folder))
                    elif os.path.isdir(src):
                        move_dir(src, os.path.join(ROOT, folder))
    print('整理完成！')
