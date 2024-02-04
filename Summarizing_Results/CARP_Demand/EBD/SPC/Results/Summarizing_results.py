import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Summary_Branch_Cut_SPC_EBD.csv')
df = df.dropna() # This line will remove the no solution rows from the dataframe
df['No. of Instances']=27
df['No. of Infeasible Instances'] = df['No. of Instances'] - df.groupby('No. of Nodes')['No. of Nodes'].transform('count')
# Getting df with Exceeded Time Limit
condition=df['Sol_Status_SP_Const']=='time limit exceeded'
condition_df=df[condition]
Time_limit_Ex_df = condition_df.groupby('Instance_name').agg({'Sol_Status_SP_Const':'count'})
Time_limit_Ex_df.rename(columns={'Sol_Status_SP_Const':'No. of Instances that Exceeded Time Limit'}, inplace=True)
df.reset_index(inplace=True)
Time_limit_Ex_df.reset_index(inplace=True)
df=pd.merge(df,Time_limit_Ex_df,on='Instance_name',how='left')
# Merge the original DataFrame with the grouped counts on 'No. of Nodes'
df=df[df['Sol_Status_SP_Const']!='time limit exceeded']
result = df.groupby('Instance_name').agg({'Instance_name':'first','No. of Nodes':'first','No. of Edges':'first','No. of Instances':'first','No. of Infeasible Instances':'first','No. of Instances that Exceeded Time Limit':'first','Total_Time_SP': 'mean'})
result = result.reset_index(drop=True)
result['NBV']=(result['No. of Nodes']*result['No. of Edges'])+result['No. of Nodes']
result.rename(columns={'No. of Nodes': '$|V|$', 'No. of Edges': '$|E|$','Instance_name':'RN Name', 'Total_Time_SP':'EBD-SPC'}, inplace=True)
result = result.sort_values(by='NBV')
result['RN No.']=range(1, len(result) + 1)
result['RN Name'] = result['RN Name'].str.replace('CARP_','').str.replace('_graph.dat', '').str.split('_').apply(lambda x: '\\_'.join(x))
results_final_1=result[['RN No.','EBD-SPC']]
pd.options.display.float_format = '{:.2f}'.format
# Convert DataFrame to LaTeX code

results_final_2=result[['RN No.','EBD-SPC','No. of Instances','No. of Infeasible Instances','No. of Instances that Exceeded Time Limit']]
latex_code = results_final_2.to_latex(index=False, escape=False,formatters={'EBD-SPC': '{:,.0f}'.format})
with open('dataframe_table_1.tex', 'w') as file:
    file.write(latex_code)

