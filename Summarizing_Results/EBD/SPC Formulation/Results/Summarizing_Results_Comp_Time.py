import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Summary_Branch_Cut_SPC_EBD.csv')
df['RN Name'] = df['Instance_name'].str.replace('CARP_','').str.replace('_graph.dat', '').str.split('_').apply(lambda x: '\\_'.join(x))
df_infeasible=df[df['Sol_Status_SP_Const']=='CPLEX Error  1217: No solution exists.']
Count_Infeasible=df_infeasible.groupby(['RN Name','tolerance']).agg({'RN Name':'first','tolerance':'first','Sol_Status_SP_Const':'count'})
Count_Infeasible.rename(columns={'Sol_Status_SP_Const':'No. of Infeasible Instances'}, inplace=True)

df_time_limit=df[df['Sol_Status_SP_Const']=='time limit exceeded']
df_time_limit_grouped=df_time_limit.groupby(['RN Name','tolerance']).agg({'RN Name':'first','tolerance':'first','Sol_Status_SP_Const':'count'})
df_time_limit_grouped.rename(columns={'Sol_Status_SP_Const':'No. of Instance that Exceeded Time Limit'}, inplace=True)


gap_analysis = df_time_limit.groupby(['RN Name', 'tolerance']).agg(mean_gap=('gap', 'mean'), max_gap=('gap', 'max')).reset_index()
gap_analysis.to_csv('gap.csv',index=False)


df_feasible_time_not_exceeded=df[(df['Sol_Status_SP_Const']!='time limit exceeded') & (df['Sol_Status_SP_Const']!='CPLEX Error  1217: No solution exists.')]
Average_by_District = df_feasible_time_not_exceeded.groupby(['RN Name','No. of Districts']).agg({'RN Name':'first','No. of Districts':'first','Total_Time_SP':'mean'})
Average_by_Tolerance = df_feasible_time_not_exceeded.groupby(['RN Name','tolerance']).agg({'RN Name':'first','tolerance':'first','Total_Time_SP':'mean'})
Average_by_Tolerance=Average_by_Tolerance[['Total_Time_SP']]
Count_Infeasible=Count_Infeasible[['No. of Infeasible Instances']]
df_time_limit_grouped=df_time_limit_grouped[['No. of Instance that Exceeded Time Limit']]
Average_by_Tolerance.reset_index(inplace=True)
Count_Infeasible.reset_index(inplace=True)
df_time_limit_grouped.reset_index(inplace=True)
df_tolerance=pd.merge(Average_by_Tolerance,Count_Infeasible,on=['RN Name','tolerance'],how='left')
df_tolerance=pd.merge(df_tolerance,df_time_limit_grouped,on=['RN Name','tolerance'],how='left')
df_tolerance['No. of Infeasible Instances'] = df_tolerance['No. of Infeasible Instances'].fillna(0)
df_tolerance['No. of Instance that Exceeded Time Limit'] = df_tolerance['No. of Instance that Exceeded Time Limit'].fillna(0)
df_tolerance['RN No.']=df_tolerance['RN Name'].apply(lambda x: 4 if x=='F6\_p' else 5)
result_final=df_tolerance[['RN No.','tolerance','Total_Time_SP','No. of Infeasible Instances','No. of Instance that Exceeded Time Limit']]
latex_code = result_final.to_latex(index=False, escape=False,formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format, 'No. of Infeasible Instances': '{:,.0f}'.format,'No. of Instance that Exceeded Time Limit': '{:,.0f}'.format})

with open('dataframe_table_2.tex', 'w') as file:
    file.write(latex_code)

Average_by_District['RN No.']=Average_by_District['RN Name'].apply(lambda x: 4 if x=='F6\_p' else 5)
result_final_1=Average_by_District[['RN No.','No. of Districts','Total_Time_SP']]


RN=[4,5]
condition1 = result_final_1['RN No.'] == 4
condition2 = result_final_1['RN No.'] == 5
fig, ax = plt.subplots()
# Plot for condition1
plt.plot(result_final_1[condition1]['No. of Districts'],result_final_1[condition1]['Total_Time_SP'],linestyle='-', color='blue',label=r'RN '+str(RN[0]))
plt.plot(result_final_1[condition2]['No. of Districts'],result_final_1[condition2]['Total_Time_SP'],linestyle='--', color='green',label=r'RN '+str(RN[1]))
plt.xlabel(r'$p$')
plt.ylabel('Average Computational Time (in Seconds)')
ax.legend()
plt.savefig('Average_Comp.png')
# Show the plots
plt.show()



gap_analysis['RN No.']=gap_analysis['RN Name'].apply(lambda x: 4 if x=='F6\_p' else 5)
gap_analysis=gap_analysis[['RN No.','tolerance','mean_gap','max_gap']]
gap_analysis['tolerance']=gap_analysis['tolerance']*100
latex_code_2 = gap_analysis.to_latex(index=False, escape=False,formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format, 'max_gap': '{:,.2f}'.format,'mean_gap': '{:,.2f}'.format})

with open('dataframe_table_3.tex', 'w') as file:
    file.write(latex_code_2)
