import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from save_load import load_parameters
from furnace_model import NormalizedFurnaceModel


# 读取CSV文件
df = pd.read_csv('R2_1200_1e-3_raw.csv')

variables = ['T', 't', 'fs', 'fl', 'x', 'y', 'w', 'rhob', 'p']

case_name = "my_design"
params = load_parameters(case_name)
model = NormalizedFurnaceModel(params)

R1 = model.ReactionRate_1(df['z'].values, df['T'].values, df['t'].values, df['fs'].values, df['x'].values, df['y'].values, df['w'].values, df['p'].values)
R2 = model.ReactionRate_2(df['z'].values, df['T'].values, df['t'].values, df['fs'].values, df['x'].values, df['y'].values, df['w'].values, df['p'].values)
R4 = model.ReactionRate_4(df['z'].values, df['T'].values, df['t'].values, df['fl'].values, df['x'].values, df['y'].values, df['w'].values, df['p'].values)
R5 = model.ReactionRate_5(df['z'].values, df['T'].values, df['t'].values, df['fs'].values, df['x'].values, df['y'].values, df['w'].values, df['p'].values)
R6 = model.ReactionRate_6(df['z'].values, df['T'].values, df['t'].values, df['x'].values, df['y'].values, df['w'].values, df['p'].values)
R7 = model.ReactionRate_7(df['T'].values, df['x'].values, df['y'].values, df['w'].values,df['p'].values)

plt.plot(df['z'], R1, label='R1')
plt.plot(df['z'], R2, label='R2')
plt.plot(df['z'], R4, label='R4')
plt.plot(df['z'], R5, label='R5')
plt.plot(df['z'], R6, label='R6')
plt.plot(df['z'], R7, label='R7')
plt.legend()
plt.show()