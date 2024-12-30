import pickle

# probs_list=['CARP_F11_g_graph.dat','CARP_F14_g_graph.dat','CARP_F6_p_graph.dat','CARP_F1_p_graph.dat'] #Problem Instances
# no_dist_list=[2,10,50,100] # the number of districts
#tol_list=[0.01,0.05,0.10,0.20,0.30,1] #Balance Tolerance

model=['EBD_SP_Cut_Empty']
# probs_list=['CARP_F15_g_graph.dat','CARP_N17_g_graph.dat','CARP_K17_g_graph.dat','CARP_K13_g_graph.dat','CARP_O17_g_graph.dat','CARP_N13_g_graph.dat']
probs_list=['CARP_N17_g_graph.dat']
no_dist_list=[10,20]
tol_list=[0.1]

task_list=[[m,k,i,j] for m in model for k in probs_list for i in no_dist_list for j in tol_list]
print(len(task_list))


task_dict={}
for i,j in enumerate(task_list):
    task_dict[i+1]=j[:]

print(task_dict)
with open("pickle_file.pykl","wb") as f:
    pickle.dump(task_dict,f)

with open("pickle_file.pykl","rb") as f2:
    read_dict=pickle.load(f2)