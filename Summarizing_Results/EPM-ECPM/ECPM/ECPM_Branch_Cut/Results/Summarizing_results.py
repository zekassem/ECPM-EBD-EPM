import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Summary_Branch_Cut_SPC_ECPM.csv')
df_1=pd.read_csv('SPC_Only_Summary.csv')
df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')
df = df.dropna()
result = df.groupby('Instance_name').agg({'Instance_name':'first','No. of Nodes':'first','No. of Edges':'first','Branch_Cut_Time': 'mean', 'Total_Time_SP': 'mean'})
result = result.reset_index(drop=True)

result['Improvement Ratio']=result['Branch_Cut_Time']/result['Total_Time_SP']

result['NBV']=(result['No. of Nodes']*result['No. of Edges'])+result['No. of Nodes']

result.rename(columns={'No. of Nodes': '$|V|$', 'No. of Edges': '$|E|$','Instance_name':'RN Name'}, inplace=True)
result = result.sort_values(by='NBV')

result = pd.concat([result, df_1], ignore_index=True)

result['RN No.']=range(1, len(result) + 1)
result['RN Name'] = result['RN Name'].str.replace('CARP_','').str.replace('_graph.dat', '').str.split('_').apply(lambda x: '\\_'.join(x))

results_final=result[['RN No.','RN Name','$|V|$','$|E|$','NBV','Branch_Cut_Time','Total_Time_SP','Improvement Ratio']]

mean_value = results_final['Improvement Ratio'].mean(skipna=True)
median_value = results_final['Improvement Ratio'].median(skipna=True)

print(mean_value)
print(median_value)

pd.options.display.float_format = '{:.2f}'.format
# Convert DataFrame to LaTeX code
latex_code = results_final.to_latex(index=False, escape=False,na_rep='-',formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'NBV': '{:,.0f}'.format, 'Branch_Cut_Time': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format,'Improvement Ratio': '{:,.2f}'.format})
results_final.to_csv('Test.csv', index=False)

# Save LaTeX code to a file
with open('dataframe_table.tex', 'w') as file:
    file.write(latex_code)


RN=[1,5,9,12]
condition1 = df['Instance_name'] == 'CARP_F15_g_graph.dat'
condition2 = df['Instance_name'] == 'CARP_O12_g_graph.dat'
condition3 = df['Instance_name'] == 'CARP_S11_g_graph.dat'
condition4 = df['Instance_name'] == 'CARP_N15_g_graph.dat'
#
fig, ax = plt.subplots()
# Plot for condition1
plt.plot(df[condition1]['No. of Districts'],df[condition1]['Speedup'],linestyle='-', color='blue',label=r'RN '+str(RN[0]))
plt.plot(df[condition2]['No. of Districts'],df[condition2]['Speedup'],linestyle='--', color='green',label=r'RN '+str(RN[1]))
plt.plot(df[condition3]['No. of Districts'],df[condition3]['Speedup'],linestyle=':', color='red',label=r'RN '+str(RN[2]))
plt.plot(df[condition4]['No. of Districts'],df[condition4]['Speedup'],linestyle='-.', color='purple',label=r'RN '+str(RN[3]))
plt.xlabel(r'$p$')
plt.ylabel('SPC/CSC Improvement Ratio (SPC/CSC)')
ax.legend()
plt.savefig('IR.png')
# Show the plots
plt.show()