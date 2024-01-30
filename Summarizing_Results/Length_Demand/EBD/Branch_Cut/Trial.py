# Newly calculated Instances
task_list=[['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 4, 0.01], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 8, 0.01], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 10, 0.01], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 10, 0.1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 20, 0.01], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 30, 0.1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 40, 0.1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 50, 0.1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 4, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 6, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 8, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 8, 0.1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 10, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 10, 0.1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 20, 0.1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 40, 0.1]]
# Infeasible Instances
task_list_2=[['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 30, 0.01], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 40, 0.01], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 50, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 20, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 30, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 30, 0.1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 40, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 50, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 50, 0.1]]
# Originally added Instances
task_list_3=[['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 2, 0.01], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 2, 0.1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 2, 1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 4, 0.1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 4, 1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 6, 0.01], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 6, 0.1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 6, 1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 8, 0.1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 8, 1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 10, 1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 20, 0.1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 20, 1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 30, 1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 40, 1], ['EBD_SP_Cut_Empty', 'CARP_F6_p_graph.dat', 50, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 2, 0.01], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 2, 0.1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 2, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 4, 0.1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 4, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 6, 0.1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 6, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 8, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 10, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 20, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 30, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 40, 1], ['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 50, 1]]
final_task_list=task_list+task_list_2+task_list_3
print(len(final_task_list))

model=['EBD_SP_Cut_Empty']
probs_list_1=['CARP_F6_p_graph.dat','CARP_O12_g_graph.dat']
no_dist_list_1=[2,4,6,8,10,20,30,40,50]
tol_list_1=[0.01,0.1,1]

task_list_trial=[[m,k,i,j] for m in model for k in probs_list_1 for i in no_dist_list_1 for j in tol_list_1]

set1 = set(map(tuple, final_task_list))
set2 = set(map(tuple, task_list_trial))


if set1 == set2:
    print("The sets are equal.")
else:
    print("The sets are not equal.")






