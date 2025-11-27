import pandas as pd

df = pd.read_csv('Summary_SPC_ECPM_only.csv')
df = df.dropna()
result = df.groupby('Instance_name').agg({'Instance_name':'first','No. of Nodes':'first','No. of Edges':'first', 'Total_Time_SP': 'mean'})
result = result.reset_index(drop=True)
result['NBV']=(result['No. of Nodes']*result['No. of Edges'])+result['No. of Nodes']
result.rename(columns={'No. of Nodes': '$|V|$', 'No. of Edges': '$|E|$','Instance_name':'RN Name'}, inplace=True)
result = result.sort_values(by='NBV')
result['RN No.']=range(1, len(result) + 1)
result['RN Name'] = result['RN Name'].str.replace('CARP_','').str.replace('_graph.dat', '').str.split('_').apply(lambda x: '\\_'.join(x))
results_final=result[['RN No.','RN Name','$|V|$','$|E|$','NBV','Total_Time_SP']]
pd.options.display.float_format = '{:.2f}'.format
results_final.to_csv('SPC_Only_Summary.csv', index=False)
