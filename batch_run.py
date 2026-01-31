# batch_run.py
import pandas as pd
from parameters import FurnaceParameters, quick_modify, create_standard_case
from furnace_model import FurnaceModel
from save_load import save_parameters

def run_batch_study(study_type="heat_loss_hp"):
    """批量运行参数研究"""
    
    base_params = create_standard_case("default")
    all_results = []
    
    if study_type == "heat_loss_hp":
        # 热损失换热系数敏感性分析
        heat_loss_hp = [8, 9, 10, 11, 12]
        
        for hp in heat_loss_hp:
            params = quick_modify(base_params, 
                                case_name=f"U_{hp}",
                                U=hp)
            
            model = FurnaceModel(params)
            results = model.run()
            
            result_info = {
                'case_name': params.case_name,
                'U': hp,
                'T_out': results['T_out'],
                't_out': results['t_out'],
                'fs_out': results['fs_out']
            }
            all_results.append(result_info)
            
            # 保存这个算例
            save_parameters(params)
    
    elif study_type == "fraction_void":
        # 床层孔隙率敏感性分析
        fraction_void = [0.20, 0.22, 0.24, 0.26, 0.28]
        
        for ep in fraction_void:
            params = quick_modify(base_params,
                                case_name=f"epsilon_{ep}",
                                epsilon=ep)
            
            model = FurnaceModel(params)
            results = model.run()
            
            result_info = {
                'case_name': params.case_name,
                'epsilon': ep,
                'p_bottom': results['p_bottom'],
            }
            all_results.append(result_info)
            save_parameters(params)

    elif study_type == "grid_independence":
        # 床层孔隙率敏感性分析
        grid_list = [500, 1000, 2000, 3000, 4000, 5000]
        
        for grid in grid_list:
            params = quick_modify(base_params,
                                case_name=f"initial_grid_{grid}",
                                initial_mesh=grid)
            
            model = FurnaceModel(params)
            results = model.run()
            
            result_info = {
                'case_name': params.case_name,
                'initial_mesh': grid,
                'T_out': results['T_out'],
                't_out': results['t_out'],
                'fs_out': results['fs_out'],
                'x_out' : results['x_out'],
                'y_out' : results['y_out']
            }
            all_results.append(result_info)
            save_parameters(params)
    
    # 保存结果汇总
    df = pd.DataFrame(all_results)
    df.to_csv(f'cases/batch_results_{study_type}.csv', index=False, encoding='utf-8')
    
    print(f"\n完成 {len(all_results)} 个算例")
    print(f"结果已保存: cases/batch_results_{study_type}.csv")
    
    return df
