import numpy as np
import csv
## End of Data
# Input parameters to the districting problem
no_dist=[2,10,30,40,50,100] # the number of districts
probs=['CARP_F15_g_graph.dat','CARP_N17_g_graph.dat','CARP_K13_g_graph.dat','CARP_O12_g_graph.dat','CARP_S9_p_graph.dat','CARP_N16_g_graph.dat','CARP_S9_g_graph.dat','CARP_S11_g_graph.dat','CARP_K9_p_graph.dat','CARP_N9_g_graph.dat','CARP_N11_g_graph.dat','CARP_N15_g_graph.dat','CARP_F6_p_graph.dat']


with open('Results/Summary_EPM.csv', 'w') as newFile:
    newFileWriter = csv.writer(newFile, lineterminator='\n')
    newFileWriter.writerow(
        ['Instance_no','Instance_name', 'No. of Nodes', 'No. of Edges', 'No. of Districts', 'No. of Threads', 'EPM_Time', 'Objective Function_EPM', 'Sol_Status_EPM'])
    i=1
    original_probs=[]
    printed_probs=[]
    for prob in probs:
        graph_file = prob
        no_nodes = np.loadtxt(graph_file, skiprows=1, max_rows=1, usecols=1,
                                   dtype=int)  # skipping lines to get the number of nodes in the planar graph
        no_nodes1 = int(no_nodes)
        no_edges = np.loadtxt(graph_file, skiprows=2, max_rows=1, usecols=1,
                                   dtype=int)  # skipping lines to get the
        no_edges1 = int(no_edges)

        for num in no_dist:
            results_file = 'EPM_no_dist_'+str(num)+'_prob_'+str(prob)+'.csv'
            original_probs.append(results_file)
            num_threads = np.loadtxt(results_file, skiprows=1, max_rows=1, usecols=0, dtype=str, delimiter=',')
            Computation_Time_for_EPM=np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=0, dtype=float, delimiter=',')
            Objective_Function_EPM = np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=1, dtype=float,
                                                         delimiter=',')
            Sol_Status_EPM=np.loadtxt(results_file, skiprows=5, max_rows=1, usecols=2, dtype=str,
                                                         delimiter=',')


            newFileWriter.writerow([i,prob, no_nodes1, no_edges1, num,num_threads, Computation_Time_for_EPM,Objective_Function_EPM, Sol_Status_EPM])
            i+=1










