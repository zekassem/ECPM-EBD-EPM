import numpy as np
import csv
## End of Data
# Input parameters to the districting problem
probs_list=['CARP_F15_g_graph.dat','CARP_N17_g_graph.dat']
no_dist_list=[2,4,6,8,10,20,30,40,50]
tol_list=[0.01,0.1,1]


with open('Results/Summary_SPC_Only.csv', 'w') as newFile:
    newFileWriter = csv.writer(newFile, lineterminator='\n')
    newFileWriter.writerow(
        ['Instance_no','Instance_name', 'No. of Nodes', 'No. of Edges', 'No. of Districts', 'tolerance','No. of Threads',  'Time_SPC','Time_print_path' ,'Total_Time','Objective_Function_SPC','Sol_Status_SPC','gap'])
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

        extra_time_file = 'time_print_path_' + str(prob) + '.csv'
        extra_time = np.loadtxt(extra_time_file, skiprows=1, max_rows=1, usecols=0,
                                dtype=float)

        for num in no_dist_list:
            for tol in tol_list:
                original_list.append('EBD_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) + '_tol_' + str(tol) + '_prob_' + str(prob) + '.csv')
                extra_time = np.loadtxt(extra_time_file, skiprows=1, max_rows=1, usecols=0,
                                        dtype=float)
                results_file = 'EBD_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) + '_tol_' + str(tol) + '_prob_' + str(
                    prob) + '.csv'

                if [prob, num, tol] == ['CARP_N17_g_graph.dat', 30, 0.1] or [prob, num, tol] == ['CARP_N17_g_graph.dat', 40, 0.1]:
                    num_threads = np.loadtxt(results_file, skiprows=1, max_rows=1, usecols=0, dtype=str, delimiter=',')

                    Time_SPC = np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=0, dtype=float,
                                          delimiter=',')
                    Total_Time_SPC = Time_SPC + extra_time
                    Objective_Function_SPC = np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=1,
                                                        dtype=float,
                                                        delimiter=',')

                    Sol_Status_SPC = np.loadtxt(results_file, skiprows=5, max_rows=1, usecols=2, dtype=str,
                                                delimiter=',')
                    gap = round(np.loadtxt(results_file, skiprows=5, max_rows=1, usecols=1, dtype=float,
                                           delimiter=',') * 100, 2)

                    newFileWriter.writerow(
                        [i, prob, no_nodes1, no_edges1, num, tol, num_threads, Time_SPC, extra_time, Total_Time_SPC,
                         Objective_Function_SPC, Sol_Status_SPC, gap])
                    i+=1
                    continue
                if [prob, num, tol] == ['CARP_N17_g_graph.dat', 40, 0.01]:
                    num_threads=Time_SPC=extra_time=Total_Time_SPC=Objective_Function_SPC=Sol_Status_SPC=gap=np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=0, dtype=str,
                                                delimiter=',')
                    newFileWriter.writerow(
                        [i, prob, no_nodes1, no_edges1, num, tol, num_threads, Time_SPC, extra_time, Total_Time_SPC,
                         Objective_Function_SPC, Sol_Status_SPC, gap])
                    i += 1
                    continue

                num_threads = np.loadtxt(results_file, skiprows=1, max_rows=1, usecols=0, dtype=str, delimiter=',')
                Sol_Status_BC = np.loadtxt(results_file, skiprows=3, max_rows=1, usecols=0, dtype=str,
                                            delimiter=',')
                Sol_Status_SPC=np.loadtxt(results_file, skiprows=5, max_rows=1, usecols=0, dtype=str,
                                            delimiter=',')
                if Sol_Status_BC=='CPLEX Error' and Sol_Status_SPC!='CPLEX Error':
                    Time_SPC = np.loadtxt(results_file, skiprows=6, max_rows=1, usecols=0, dtype=float,
                                          delimiter=',')
                    Total_Time_SPC = Time_SPC + extra_time
                    Objective_Function_SPC = np.loadtxt(results_file, skiprows=6, max_rows=1, usecols=1,
                                                        dtype=float,
                                                        delimiter=',')

                    Sol_Status_SPC = np.loadtxt(results_file, skiprows=8, max_rows=1, usecols=2, dtype=str,
                                                delimiter=',')
                    gap = round(np.loadtxt(results_file, skiprows=8, max_rows=1, usecols=1, dtype=float,
                                           delimiter=',') * 100, 2)
                    newFileWriter.writerow([i, prob, no_nodes1, no_edges1, num, tol, num_threads, Time_SPC, extra_time, Total_Time_SPC,Objective_Function_SPC, Sol_Status_SPC, gap])
                    i += 1

                elif Sol_Status_BC=='CPLEX Error' and Sol_Status_SPC=='CPLEX Error':
                    list_probs.append(results_file)
                    Total_Time_SPC = extra_time = gap = Sol_Status_SPC = Objective_Function_SPC = Time_SPC = np.loadtxt(
                        results_file, skiprows=6, max_rows=1, usecols=0, dtype=str, delimiter=',')
                    newFileWriter.writerow(
                        [i, prob, no_nodes1, no_edges1, num, tol, num_threads, Time_SPC, extra_time, Total_Time_SPC,
                         Objective_Function_SPC, Sol_Status_SPC, gap])
                    i += 1
                else:
                    Time_SPC = np.loadtxt(results_file, skiprows=7, max_rows=1, usecols=0, dtype=float,
                                          delimiter=',')
                    Total_Time_SPC = Time_SPC + extra_time
                    Objective_Function_SPC = np.loadtxt(results_file, skiprows=7, max_rows=1, usecols=1,
                                                        dtype=float,
                                                        delimiter=',')

                    Sol_Status_SPC = np.loadtxt(results_file, skiprows=9, max_rows=1, usecols=2, dtype=str,
                                                delimiter=',')
                    gap = round(np.loadtxt(results_file, skiprows=9, max_rows=1, usecols=1, dtype=float,
                                           delimiter=',') * 100, 2)
                    newFileWriter.writerow(
                        [i, prob, no_nodes1, no_edges1, num, tol, num_threads, Time_SPC, extra_time, Total_Time_SPC,
                         Objective_Function_SPC, Sol_Status_SPC, gap])
                    i += 1







print(list_probs)
print(len(list_probs))




