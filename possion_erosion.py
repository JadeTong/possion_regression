'''
泊松回归是一种用于建模计数数据的回归分析方法，
特别适用于响应变量是计数（如事件发生的次数）且服从泊松分布的情况。常见应用包括交通事故发生次数、疾病病例数量等。
1）计数数据
2）事件独立性（每个观察单位的事件发生是独立的）
3）因变量均值与方差相等
4）数据为稀疏或离散的事件（某段时间内某个地区发生的罕见事件）
5）时间或空间固定的观察（单位时间内或单位面积内的事件发生次数）
6）考虑协变量的影响：泊松回归可以通过引入协变量（自变量）来研究其对计数响应变量的影响，允许我们建模与时间、地点或其他条件相关的计数数据。

泊松回归——航班数据分析
1、数据及分析对象
'o-ring-erosion-only.csv'，数据内容来自于UCI数据集中的Challenger USA Space Shuttle O-Ring Data Set（1993）。
该数据集给出了美国23次航天飞机的飞行数据。主要属性如下：
Number of O-rings at risk on a given flight：航班上存在潜在风险的O型环数量；
Number experiencing thermal distress：出现热损伤的O型环数量；
Launch temperature (degrees F)：发射温度（华氏度）；
Leak-check pressure (psi)：检漏压力（psi）；
Temporal order of flight：航班时序。

2、目的及分析任务
理解机器学习方法在数据分析中的应用——采用泊松回归方法进行回归分析：
（1）以全部记录为训练集进行泊松回归建模；
（2）对模型进行假设检验和可视化处理，验证泊松回归建模的有效性；
（3）按训练模型预测训练集并得到均方根误差。

3.方法及工具
pandas, numpy, statsmodels
'''

#%%                  1.业务理解
'''
本例题所涉及的业务为分析航天飞机出现热损伤O形环的可能性。
该业务的主要内容是通过建立泊松回归模型实现依据给定发射条件预测航天飞机出现热损伤的O形环的数量。
'''

#%%                  2.数据读取
import pandas as pd
erosion = pd.read_csv('D:/desktop/ML/回归分析/o-ring-erosion-only.csv',
                      names = ['num_rings',
                               'num_distress',
                               'temperature',
                               'pressure',
                               'order'])
#因为数据无表头，所以要手动生成
#因变量为飞行中热损伤O形环的数量num_distress，将数据框列重新排序
order = ['num_rings','temperature','pressure','order','num_distress']
erosion = erosion[order]
erosion
#%%                  3.数据理解
#%%% 探索性分析
erosion.describe()
import matplotlib.pyplot as plt
plt.hist(erosion.num_distress,bins=10)
#%%% 计算比较因变量的均值和方差
import numpy as np
print(np.mean(erosion.num_distress))   # 0.391304347826087
print(np.var(erosion.num_distress))    # 0.41209829867674863
#方差约等于均值，避免了在泊松回归中发生过度分散（均值小于方差）或分散不足（均值大于方差）的情况。

#%%                  4.建模
import statsmodels.api as sm
import statsmodels.formula.api as smf
X = erosion[['num_rings', 'temperature', 'pressure', 'order']]
Y = erosion.num_distress
glm = smf.glm('num_distress ~ num_rings + temperature + pressure + order', erosion, family=sm.families.Poisson())
result = glm.fit()
print(result.summary())
# =============================================================================
#                  Generalized Linear Model Regression Results                  
# ==============================================================================
# Dep. Variable:           num_distress   No. Observations:                   23
# Model:                            GLM   Df Residuals:                       19
# Model Family:                 Poisson   Df Model:                            3
# Link Function:                    Log   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -15.317
# Date:                Wed, 30 Oct 2024   Deviance:                       15.407
# Time:                        02:56:36   Pearson chi2:                     23.4
# No. Iterations:                     5   Pseudo R-squ. (CS):             0.2633
# Covariance Type:            nonrobust                                         
# ===============================================================================
#                   coef    std err          z      P>|z|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept       0.0984      0.090      1.094      0.274      -0.078       0.275
# num_rings       0.5905      0.540      1.094      0.274      -0.468       1.649
# temperature    -0.0883      0.042     -2.092      0.036      -0.171      -0.006
# pressure        0.0070      0.010      0.708      0.479      -0.012       0.026
# order           0.0115      0.077      0.150      0.881      -0.138       0.161
# ===============================================================================

#只有temperature的P值小于0.05，说明其它解释变量在控制temperature的情况下，对因变量的影响不显著。

#%%                  5.模型评价
#用均方根误差RMSE，来评估模型预测结果
Y_pred = result.predict(X)
# 计算 RMSE
from sklearn.metrics import mean_squared_error
RMSE = np.sqrt(mean_squared_error(Y_pred, Y))
print("RMSE:", RMSE)
# RMSE: 0.4895957413435296




















