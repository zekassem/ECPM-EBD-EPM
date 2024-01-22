import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Summary_Branch_Cut_SPC_ECPM.csv')
df_1 = pd.read_csv('Summary_EPM.csv')
df_2=pd.read_csv('SPC_Only_Summary.csv')
df=pd.merge(df,df_1,on=['No. of Nodes','No. of Districts'], how='left',suffixes=('', '_df2'))
df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')
df = df.dropna()
df.to_csv('SPC_EPM.csv', index=False)
result = df.groupby('Instance_name').agg({'Instance_name':'first','No. of Nodes':'first','No. of Edges':'first', 'EPM_Time':'mean', 'Total_Time_SP': 'mean'})
result = result.reset_index(drop=True)
result['Improvement Ratio']=result['EPM_Time']/result['Total_Time_SP']
result['NBV']=(result['No. of Nodes']*result['No. of Edges'])+result['No. of Nodes']
result.rename(columns={'No. of Nodes': '$|V|$', 'No. of Edges': '$|E|$','Instance_name':'RN Name'}, inplace=True)
result = result.sort_values(by='NBV')
result = pd.concat([result, df_2], ignore_index=True)
result['RN No.']=range(1, len(result) + 1)
result['RN Name'] = result['RN Name'].str.replace('CARP_','').str.replace('_graph.dat', '').str.split('_').apply(lambda x: '\\_'.join(x))
results_final=result[['RN No.','RN Name','$|V|$','$|E|$','NBV','EPM_Time','Total_Time_SP','Improvement Ratio']]
pd.options.display.float_format = '{:.2f}'.format
results_g_1=results_final[results_final['Improvement Ratio']>1]
results_g_1.to_csv('Test.csv', index=False)
mean_value = results_g_1['Improvement Ratio'].mean(skipna=True)
median_value = results_g_1['Improvement Ratio'].median(skipna=True)
print(mean_value)
print(median_value)

# Convert DataFrame to LaTeX code
latex_code = results_final.to_latex(index=False, escape=False,na_rep='-',formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'NBV': '{:,.0f}'.format, 'Branch_Cut_Time': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format,'EPM_Time': '{:,.0f}'.format,'Improvement Ratio': '{:,.2f}'.format})
results_final.to_csv('Test.csv', index=False)

# Save LaTeX code to a file
with open('dataframe_table.tex', 'w') as file:
    file.write(latex_code)