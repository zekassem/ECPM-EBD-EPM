import numpy as np
import csv
## End of Data
# Input parameters to the districting problem
no_dist=[2,10,30,40,50,100] # the number of districts
probs=['CARP_K9_g_graph.dat']


with open('Results/Summary_SPC_ECPM_only.csv', 'w') as newFile:
    newFileWriter = csv.writer(newFile, lineterminator='\n')
    newFileWriter.writerow(
        ['Instance_no','Instance_name', 'No. of Nodes', 'No. of Edges', 'No. of Districts', 'No. of Threads', 'Time_SP_Const', 'Time_to_print_path','Total_Time_SP','Objective_Function_SPC','Sol_Status_SP_Const'])
    i=1
    for prob in probs:
        graph_file = prob
        no_nodes = np.loadtxt(graph_file, skiprows=1, max_rows=1, usecols=1,
                                   dtype=int)  # skipping lines to get the number of nodes in the planar graph
        no_nodes1 = int(no_nodes)
        no_edges = np.loadtxt(graph_file, skiprows=2, max_rows=1, usecols=1,
                                   dtype=int)  # skipping lines to get the
        no_edges1 = int(no_edges)

        extra_time_file='time_print_path_'+str(prob)+'.csv'
        extra_time=np.loadtxt(extra_time_file, skiprows=1, max_rows=1, usecols=0,
                                   dtype=float)


        for num in no_dist:
            results_file = 'ECPM_Cut_Set_vs_SP_Contiguity_no_dist_'+str(num)+'_prob_'+str(prob)+'.csv'
            num_threads = np.loadtxt(results_file, skiprows=1, max_rows=1, usecols=0, dtype=str, delimiter=',')
            Time_SP_Const=np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=0, dtype=float,
                                                         delimiter=',')
            Total_Time_SP=Time_SP_Const+extra_time
            Objective_Function_SPC = np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=1, dtype=float,
                                                delimiter=',')
            Sol_Status_SP_Const = np.loadtxt(results_file, skiprows=5, max_rows=1, usecols=2, dtype=str,
                                       delimiter=',')

            newFileWriter.writerow([i,prob, no_nodes1, no_edges1, num,num_threads, Time_SP_Const,extra_time,Total_Time_SP,Objective_Function_SPC,Sol_Status_SP_Const])
            i+=1










