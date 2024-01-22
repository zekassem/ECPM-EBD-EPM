import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Summary_Branch_Cut_SPC_EBD.csv')
df['RN Name'] = df['Instance_name'].str.replace('CARP_','').str.replace('_graph.dat', '').str.split('_').apply(lambda x: '\\_'.join(x))
df_infeasible=df[df['Sol_Status_SP_Const']=='CPLEX Error  1217: No solution exists.']
Count_Infeasible=df_infeasible.groupby(['RN Name','tolerance']).agg({'RN Name':'first','tolerance':'first','Sol_Status_SP_Const':'count'})
Count_Infeasible.rename(columns={'Sol_Status_SP_Const':'No. of Infeasible Instances'}, inplace=True)

df_time_limit=df[df['Sol_Status_SP_Const']=='time limit exceeded']
gap_analysis=df_infeasible.groupby(['RN Name','tolerance']).agg({'gap':['mean','max']})
gap_analysis.rename(columns={'gap':'Gap'}, inplace=True)

print(df_time_limit)


df_feasible_time_not_exceeded=df[(df['Sol_Status_SP_Const']!='time limit exceeded') & (df['Sol_Status_SP_Const']!='CPLEX Error  1217: No solution exists.')]
df['Total_Time_SP'] = pd.to_numeric(df['Total_Time_SP'], errors='coerce')
Average_by_District = df_feasible_time_not_exceeded.groupby(['RN Name','No. of Districts']).agg({'RN Name':'first','No. of Districts':'first','Total_Time_SP':'mean'})
Average_by_Tolerance = df_feasible_time_not_exceeded.groupby(['RN Name','tolerance']).agg({'RN Name':'first','tolerance':'first','Total_Time_SP':'mean'})
Average_by_Tolerance=Average_by_Tolerance[['Total_Time_SP']]
Count_Infeasible=Count_Infeasible[['No. of Infeasible Instances']]
Average_by_Tolerance.reset_index(inplace=True)
Count_Infeasible.reset_index(inplace=True)
df_tolerance=pd.merge(Average_by_Tolerance,Count_Infeasible,on=['RN Name','tolerance'],how='left')
df_tolerance['No. of Infeasible Instances'] = df_tolerance['No. of Infeasible Instances'].fillna(0)
latex_code = df_tolerance.to_latex(index=False, escape=False,formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format, 'No. of Infeasible Instances': '{:,.0f}'.format,'SPC-3 EBD': '{:,.0f}'.format})

with open('dataframe_table_3.tex', 'w') as file:
    file.write(latex_code)

latex_code_2 = Average_by_Tolerance.to_latex(index=False, escape=False,formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format, 'Branch_Cut_Time': '{:,.0f}'.format,'SPC-3 EBD': '{:,.0f}'.format})
with open('dataframe_table_4.tex', 'w') as file:
    file.write(latex_code_2)




