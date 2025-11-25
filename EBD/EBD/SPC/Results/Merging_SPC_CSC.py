import pandas as pd
df = pd.read_csv('Summary_Branch_Cut_SPC_EBD.csv')
df_1 = pd.read_csv('Summary_EBD_Branch_Cut_Only.csv')
df_merged=pd.merge(df,df_1,on=['Instance_name','No. of Districts','tolerance'],how='right',suffixes=('', '_df2'))
df_merged.to_csv('merge.csv',index=False)
df_merged_wo_BC_Time_Limit=df_merged[df_merged['Sol_Status_B&C']!='time limit exceeded']
df_merged_wo_BC_Time_Limit['Deviation']=abs(df_merged_wo_BC_Time_Limit['Objective_Function_SPC']-df_merged_wo_BC_Time_Limit['Objective_Function_B&C'])/df_merged_wo_BC_Time_Limit['Objective_Function_SPC']
df_merged_wo_BC_Time_Limit.to_csv('dev.csv',index=False)
