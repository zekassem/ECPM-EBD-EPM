import pickle

model=['EBD_SP_Cut_Empty']
probs_list=['CARP_O12_g_graph.dat']
no_dist_list=[2,4]
tol_list=[1]
task_list=[[m,k,i,j] for m in model for k in probs_list for i in no_dist_list for j in tol_list]


task_dict={}
for i,j in enumerate(task_list):
    task_dict[i+1]=j[:]

print(task_dict)
with open("pickle_file.pykl","wb") as f:
    pickle.dump(task_dict,f)

with open("pickle_file.pykl","rb") as f2:
    read_dict=pickle.load(f2)