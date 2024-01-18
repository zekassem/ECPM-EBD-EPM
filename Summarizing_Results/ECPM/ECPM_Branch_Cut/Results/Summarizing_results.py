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
pd.options.display.float_format = '{:.2f}'.format
# Convert DataFrame to LaTeX code
latex_code = results_final.to_latex(index=False, escape=False,na_rep='-',formatters={'$|E|$': '{:,.0f}'.format,'$|V|$': '{:,.0f}'.format,'NBV': '{:,.0f}'.format, 'Branch_Cut_Time': '{:,.0f}'.format,'Total_Time_SP': '{:,.0f}'.format,'Improvement Ratio': '{:,.2f}'.format})
results_final.to_csv('Test.csv', index=False)

# Save LaTeX code to a file
with open('dataframe_table.tex', 'w') as file:
    file.write(latex_code)


# nodes=[198,761,1564,2171]
# condition1 = df['No. of Nodes'] == 198
# condition2 = df['No. of Nodes'] == 761
# condition3 = df['No. of Nodes'] == 1564
# condition4 = df['No. of Nodes'] == 2171
#
# fig, ax = plt.subplots()
# # Plot for condition1
# plt.plot(df[condition1]['No. of Districts'],df[condition1]['Speedup'],linestyle='-', color='blue',label=r'$|V|$='+str(nodes[0]))
# plt.plot(df[condition2]['No. of Districts'],df[condition2]['Speedup'],linestyle='-', color='green',label=r'$|V|$='+str(nodes[1]))
# plt.plot(df[condition3]['No. of Districts'],df[condition3]['Speedup'],linestyle='-', color='red',label=r'$|V|$='+str(nodes[2]))
# plt.plot(df[condition4]['No. of Districts'],df[condition4]['Speedup'],linestyle='-', color='purple',label=r'$|V|$='+str(nodes[3]))
# plt.xlabel(r'$p$')
# plt.ylabel('Improvement Ration (IR)')
# ax.legend()
#
# plt.savefig('IR.png')
# # Show the plots
# plt.show()