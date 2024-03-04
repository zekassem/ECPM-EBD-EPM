import pickle

model=['EBD_SP_Cut_Empty']
probs_list=['CARP_F6_p_graph.dat','CARP_O12_g_graph.dat']
no_dist_list=[2,4,6,8,10,20,30,40,50]
tol_list=[0.01,0.1,1]
task_list=[[m,k,i,j] for m in model for k in probs_list for i in no_dist_list for j in tol_list]


task_dict={}
for i,j in enumerate(task_list):
    task_dict[i+1]=j[:]

print(task_dict)

for key,value in task_dict.items():
    if value==['EBD_SP_Cut_Empty', 'CARP_O12_g_graph.dat', 2, 1]:
        print(key)

with open("pickle_file.pykl","wb") as f:
    pickle.dump(task_dict,f)

with open("pickle_file.pykl","rb") as f2:
    read_dict=pickle.load(f2)