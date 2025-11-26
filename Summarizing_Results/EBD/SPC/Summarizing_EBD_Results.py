import numpy as np
import csv
## End of Data
# Input parameters to the districting problem
probs_list=['CARP_F6_p_graph.dat','CARP_O12_g_graph.dat']
no_dist_list=[2,4,6,8,10,20,30,40,50]
tol_list=[0.01,0.1,0.5]


with open('Results/Summary_Branch_Cut_SPC_EBD.csv', 'w') as newFile:
    newFileWriter = csv.writer(newFile, lineterminator='\n')
    newFileWriter.writerow(
        ['Instance_no','Instance_name', 'No. of Nodes', 'No. of Edges', 'No. of Districts', 'tolerance','No. of Threads',  'Time_SP_Const', 'Time_to_print_path','Total_Time_SP','Objective_Function_SPC','Sol_Status_SP_Const','gap'])
    i=1
    list_probs = []
    original_list = []
    for prob in probs_list:
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


        for num in no_dist_list:
            for tol in tol_list:
                original_list.append('EBD_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) + '_tol_' + str(tol) + '_prob_' + str(prob) + '.csv')
                extra_time = np.loadtxt(extra_time_file, skiprows=1, max_rows=1, usecols=0,
                                        dtype=float)
                results_file = 'EBD_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) + '_tol_' + str(tol) + '_prob_' + str(
                    prob) + '.csv'
                print(prob,num,tol)
                Sol_Status_SP_Const=np.loadtxt(results_file, skiprows=2, max_rows=1, usecols=0, dtype=str,
                                               delimiter=',')
                if Sol_Status_SP_Const!='CPLEX Error':
                    num_threads = np.loadtxt(results_file, skiprows=1, max_rows=1, usecols=0, dtype=str, delimiter=',')
                    Time_SPC=np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=0, dtype=float,
                                                                 delimiter=',')
                    Total_Time_SPC=Time_SPC+extra_time
                    Objective_Function_SPC = np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=1, dtype=float,
                                                        delimiter=',')

                    Sol_Status_SPC = np.loadtxt(results_file, skiprows=5, max_rows=1, usecols=2, dtype=str,
                                               delimiter=',')
                    gap=round(np.loadtxt(results_file, skiprows=5, max_rows=1, usecols=1, dtype=float,
                                               delimiter=',')*100,2)

                    newFileWriter.writerow(
                        [i, prob, no_nodes1, no_edges1, num, tol, num_threads, Time_SPC, extra_time, Total_Time_SPC,
                         Objective_Function_SPC, Sol_Status_SPC, gap])
                    i+=1
                else:
                    list_probs.append(['EBD_SP_Cut_Empty',prob,num,tol])
                    Sol_Status_SPC = np.loadtxt(
                        results_file, skiprows=3, max_rows=1, usecols=0, dtype=str, delimiter=',')
                    Total_Time_SPC= extra_time = gap =Objective_Function_SPC = Time_SPC=''
                    newFileWriter.writerow(
                        [i, prob, no_nodes1, no_edges1, num, tol, num_threads, Time_SPC, extra_time, Total_Time_SPC,
                         Objective_Function_SPC, Sol_Status_SPC, gap])
                    i += 1





print(list_probs)
print(len(list_probs))




