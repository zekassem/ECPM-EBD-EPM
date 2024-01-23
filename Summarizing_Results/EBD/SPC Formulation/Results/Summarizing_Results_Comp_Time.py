import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Summary_Branch_Cut_SPC_EBD.csv')
df['RN Name'] = df['Instance_name'].str.replace('CARP_','').str.replace('_graph.dat', '').str.split('_').apply(lambda x: '\\_'.join(x))
df_infeasible=df[df['Sol_Status_SP_Const']=='CPLEX Error  1217: No solution exists.']
Count_Infeasible=df_infeasible.groupby(['RN Name','tolerance']).agg({'RN Name':'first','tolerance':'first','Sol_Status_SP_Const':'count'})
Count_Infeasible.rename(columns={'Sol_Status_SP_Const':'No. of Infeasible Instances'}, inplace=True)

df_time_limit=df[df['Sol_Status_SP_Const']=='time limit exceeded']
gap_analysis=df_time_limit.groupby(['RN Name','tolerance']).agg({'RN Name':'first','tolerance':'first','Sol_Status_SP_Const':'count'})
gap_analysis.rename(columns={'Sol_Status_SP_Const':'No. of Instance that Exceeded Time Limit'}, inplace=True)
df_time_limit.to_csv('gap.csv',index=False)



df_feasible_time_not_exceeded=df[(df['Sol_Status_SP_Const']!='time limit exceeded') & (df['Sol_Status_SP_Const']!='CPLEX Error  1217: No solution exists.')]
Average_by_District = df_feasible_time_not_exceeded.groupby(['RN Name','No. of Districts']).agg({'RN Name':'first','No. of Districts':'first','Total_Time_SP':'mean'})
Average_by_Tolerance = df_feasible_time_not_exceeded.groupby(['RN Name','tolerance']).agg({'RN Name':'first','tolerance':'first','Total_Time_SP':'mean'})
Average_by_Tolerance=Average_by_Tolerance[['Total_Time_SP']]
Count_Infeasible=Count_Infeasible[['No. of Infeasible Instances']]
gap_analysis=gap_analysis[['No. of Instance that Exceeded Time Limit']]
Average_by_Tolerance.reset_index(inplace=True)
Count_Infeasible.reset_index(inplace=True)
gap_analysis.reset_index(inplace=True)
df_tolerance=pd.merge(Average_by_Tolerance,Count_Infeasible,on=['RN Name','tolerance'],how='left')
df_tolerance=pd.merge(df_tolerance,gap_analysis,on=['RN Name','tolerance'],how='left')
df_tolerance['No. of Infeasible Instances'] = df_tolerance['No. of Infeasible Instances'].fillna(0)
df_tolerance['No. of Instance that Exceeded Time Limit'] = df_tolerance['No. of Instance that Exceeded Time Limit'].fillna(0)
df_tolerance['RN No.']=df_tolerance['RN Name'].apply(lambda x: 4 if x=='F6\_p' else 5)
result_final=df_tolerance[['RN No.','tolerance','Total_Time_SP','No. of Infeasible Instances','No. of Instance that Exceeded Time Limit']]
latex_code = result_final.to_latex(index=False, escape=False,formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format, 'No. of Infeasible Instances': '{:,.0f}'.format,'No. of Instance that Exceeded Time Limit': '{:,.0f}'.format})

with open('dataframe_table_2.tex', 'w') as file:
    file.write(latex_code)





