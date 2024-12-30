import pickle

# probs_list=['CARP_F11_g_graph.dat','CARP_F14_g_graph.dat','CARP_F6_p_graph.dat','CARP_F1_p_graph.dat'] #Problem Instances
# no_dist_list=[2,10,50,100] # the number of districts
#tol_list=[0.01,0.05,0.10,0.20,0.30,1] #Balance Tolerance

probs_list=['CARP_K9_g_graph.dat','CARP_N11_g_graph.dat','CARP_N15_g_graph.dat','CARP_N9_g_graph.dat','CARP_S11_g_graph.dat','CARP_S9_g_graph.dat']
no_dist_list=[2,10,30,40,50,100]

task_list=[[k,i] for k in probs_list for i in no_dist_list]
print(task_list)

task_dict={}
for i,j in enumerate(task_list):
    task_dict[i+1]=j[:]

print(task_dict)
with open("pickle_file.pykl","wb") as f:
    pickle.dump(task_dict,f)

with open("pickle_file.pykl","rb") as f2:
    read_dict=pickle.load(f2)