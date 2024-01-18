import numpy as np
import csv
## End of Data
# Input parameters to the districting problem
no_dist=[2,10,30,40,50,100] # the number of districts
probs=['CARP_S11_g_graph.dat','CARP_S9_p_graph.dat','CARP_S9_g_graph.dat','CARP_O12_g_graph.dat','CARP_N17_g_graph.dat','CARP_N16_g_graph.dat','CARP_N15_g_graph.dat','CARP_N11_g_graph.dat','CARP_N9_g_graph.dat','CARP_K13_g_graph.dat','CARP_K9_p_graph.dat','CARP_F15_g_graph.dat']


with open('Results/Summary_Branch_Cut_SPC_ECPM.csv', 'w') as newFile:
    newFileWriter = csv.writer(newFile, lineterminator='\n')
    newFileWriter.writerow(
        ['Instance_no','Instance_name', 'No. of Nodes', 'No. of Edges', 'No. of Districts', 'No. of Threads', 'Branch_Cut_Time', 'Objective Function_Branch_Cut', 'Sol_Status_Branch_Cut', 'Time_SP_Const', 'Time_to_print_path','Total_Time_SP','Objective_Function_SPC','Sol_Status_SP_Const', 'Speedup'])
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
            Computation_Time_for_Branch_Cut=np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=0, dtype=float, delimiter=',')
            Objective_Function_Branch_Cut = np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=1, dtype=float,
                                                         delimiter=',')
            Sol_Status_Branch_Cut=np.loadtxt(results_file, skiprows=5, max_rows=1, usecols=2, dtype=str,
                                                         delimiter=',')
            Time_SP_Const=np.loadtxt(results_file, skiprows=7, max_rows=1, usecols=0, dtype=float,
                                                         delimiter=',')
            Total_Time_SP=Time_SP_Const+extra_time
            Objective_Function_SPC = np.loadtxt(results_file, skiprows=7, max_rows=1, usecols=1, dtype=float,
                                                delimiter=',')
            Sol_Status_SP_Const = np.loadtxt(results_file, skiprows=9, max_rows=1, usecols=2, dtype=str,
                                       delimiter=',')

            Speedup=round(Computation_Time_for_Branch_Cut/Total_Time_SP,2)
            newFileWriter.writerow([i,prob, no_nodes1, no_edges1, num,num_threads, Computation_Time_for_Branch_Cut,Objective_Function_Branch_Cut, Sol_Status_Branch_Cut,Time_SP_Const,extra_time,Total_Time_SP,Objective_Function_SPC,Sol_Status_SP_Const,Speedup])
            i+=1










