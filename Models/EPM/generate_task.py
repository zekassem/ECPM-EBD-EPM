import pickle

probs_list=['CARP_F15_g_graph.dat','CARP_N17_g_graph.dat','CARP_K13_g_graph.dat','CARP_O12_g_graph.dat','CARP_S9_p_graph.dat','CARP_N16_g_graph.dat','CARP_S9_g_graph.dat','CARP_S11_g_graph.dat','CARP_K9_p_graph.dat','CARP_N9_g_graph.dat','CARP_N11_g_graph.dat','CARP_N15_g_graph.dat','CARP_K9_g_graph.dat']
no_dist_list=[2,10,30,40,50,100]


task_list=[[k,i] for k in probs_list for i in no_dist_list]
print(task_list)

task_dict={}
for i,j in enumerate(task_list):
    task_dict[i+1]=j[:]

print(task_dict)

for key,value in task_dict.items():
    if value == ['CARP_K9_g_graph.dat', 2] or value == ['CARP_K9_g_graph.dat', 10]:
        print(key)
