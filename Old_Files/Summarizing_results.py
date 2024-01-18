import pandas as pd


df = pd.read_excel('ECPM_Summary_Cut_Empty_unlimited_threads.xlsx', sheet_name='ECPM_Summary_Cut_Empty_unlimite')

result = df.groupby('Instance_name').agg({'No. of Nodes':'first','Total_Time_Cut_Set': 'mean', 'Total_Time_SP': 'mean'})

result['Improvement Ratio']=result['Total_Time_Cut_Set']/result['Total_Time_SP']

df_2=pd.read_excel('Instances.xlsx', sheet_name='Sheet1')



merged_df = pd.merge(df_2,result , right_on='No. of Nodes', left_on='$|V|$')
# merge the instances table with ECPM table

results_final=merged_df[['RN No.','RN Name','$|V|$','$|E|$','NBV','Total_Time_Cut_Set','Total_Time_SP','Improvement Ratio']]

pd.options.display.float_format = '{:.2f}'.format
# Convert DataFrame to LaTeX code
latex_code = results_final.to_latex(index=False, escape=False,formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'NBV': '{:,.0f}'.format, 'Total_Time_Cut_Set': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format,'Improvement Ratio': '{:,.2f}'.format})

print(results_final)

# Save LaTeX code to a file
with open('dataframe_table.tex', 'w') as file:
    file.write(latex_code)
