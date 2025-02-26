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

本例题所涉及的业务为分析航天飞机出现热损伤O形环的可能性。  
该业务的主要内容是通过建立泊松回归模型实现依据给定发射条件预测航天飞机出现热损伤的O形环的数量。
