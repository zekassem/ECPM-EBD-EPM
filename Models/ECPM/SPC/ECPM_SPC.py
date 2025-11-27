import time
import numpy as np
import collections
import csv
import pandas as pd
import cplex
from itertools import combinations
import sys
import traceback

class Graph():
    def __init__(self, prob):
        self.file = prob
        self.no_nodes = np.loadtxt(self.file, skiprows=1, max_rows=1, usecols=1,
                                   dtype=int)  # skipping lines to get the number of nodes in the planar graph
        self.no_nodes1 = int(self.no_nodes)
        self.no_edges = np.loadtxt(self.file, skiprows=2, max_rows=1, usecols=1,
                                   dtype=int)  # skipping lines to get the
        self.no_edges1 = int(self.no_edges)
        self.adj_list = np.loadtxt(self.file, skiprows=9, max_rows=self.no_edges1, usecols=(2, 3), dtype=int)
        self.nodes_list = list(range(1, (self.no_nodes) + 1))
        self.from_node_i = self.adj_list[:, 0].tolist()
        self.to_node_j = self.adj_list[:, 1].tolist()
        self.edges_list = list(range(1, self.no_edges + 1))  # Edges Vector
        # # Finding incident edges for every node
        self.edges = collections.defaultdict(list)
        for i in range(len(self.from_node_i)):
            self.edges[self.from_node_i[i] + 1].append(i + 1)
            self.edges[self.to_node_j[i] + 1].append(i + 1)

        self.edge_lengths = np.loadtxt(self.file, skiprows=9, max_rows=self.no_edges1, usecols=4, dtype=float)
        # This block calculates the length of each edge
        index = 0
        self.euc_dist = collections.defaultdict(dict)  # dictionary to store the Euclidean Distance of each edge, since it is a symmetric matrix we add the below two lines
        for ind, j in enumerate(self.from_node_i):
            self.euc_dist[self.from_node_i[ind] + 1][self.to_node_j[index] + 1] = self.edge_lengths[index]
            self.euc_dist[self.to_node_j[ind] + 1][self.from_node_i[index] + 1] = self.edge_lengths[index]
            index += 1

        # # creating a vector of index for every pair of node (i) and edge (e)
        self.node_i = []
        self.edge_e = []
        for i in self.nodes_list:
            for j in self.edges_list:
                self.node_i.append(i)
                self.edge_e.append(j)

        self.adj = collections.defaultdict(list)
        for i in range(len(self.from_node_i)):
            self.adj[self.from_node_i[i] + 1].append(self.to_node_j[i] + 1)
            self.adj[self.to_node_j[i] + 1].append(self.from_node_i[i] + 1)

        # # Corresponding nodes for every edge
        self.nodes_corr = collections.defaultdict(list)
        for i in range(len(self.from_node_i)):
            self.nodes_corr[i + 1].append(self.from_node_i[i] + 1)
            self.nodes_corr[i + 1].append(self.to_node_j[i] + 1)
            self.nodes_corr[i + 1] = sorted(self.nodes_corr[i + 1])

        t0 = time.time()
        self.edge_number = collections.defaultdict(list)
        for key, value in self.nodes_corr.items():
            self.edge_number[(value[0], value[1])] = key

        self.d_1 = round(time.time() - t0, 2)
        print('time to create this dictionary')
        print(self.d_1)

        # # Creating a list of the degree for every node
        self.degree = []
        for i in self.nodes_list:
            self.degree.append(len(self.edges[i]))

        # # Creating a dictionary for neighboring edges for each edge (Looks like it can be further simplified)
        self.edges_neighboring = collections.defaultdict(list)
        for e in self.edges_list:
            nodes_corr = []
            nodes_corr.append(self.from_node_i[e - 1] + 1)
            nodes_corr.append(self.to_node_j[e - 1] + 1)
            self.edges_nei = []
            for i in nodes_corr:
                self.edges_nei.extend(self.edges[i])
            self.edges_nei1 = list(set(self.edges_nei))
            l = []
            l.append(e)
            self.edges_nei = [x for x in self.edges_nei1 if x not in l]
            self.edges_neighboring[e].extend(self.edges_nei)

        self.euc_dist_edge_index = {}  # creating a dictionary that has the length of each edge using the edge index
        for index, i in enumerate(self.from_node_i):
            self.euc_dist_edge_index[index + 1] = self.edge_lengths[index]

        self.nodetoedge_net_dist = pd.read_csv("nodetoedge_distance_" + str(prob) + ".csv", header=0, index_col=0)
        self.nodetoedge_path = pd.read_csv("nodetoedge_path_" + str(prob) + ".csv", index_col=0, header=0)
        self.nodetoedge_path.columns = self.edges_list
        self.nodetoedge_path = self.nodetoedge_path.transpose()
        self.nodetoedge_path_dict = self.nodetoedge_path.to_dict()


## End of Data

# Breadth First Function
def BFS(Graph,index, result, l):
    # defining the arrays that will be in the algorithm
    pred = []  # predecessor vector
    color = []  # color vector, 0 means white, 1 means gray ,2 means black
    d = []  # distance vector
    Q = []  # set of gray vectors
    s = l  # the source edge index in the edge vector
    for e in Graph.edges_list:
        color.append(int(0))  # having all the colors of edges to be white
        d.append(int(0))
        pred.append(int(0))

    color[s - 1] = 1
    Q.append(s)
    current_dis = 0

    while len(Q) != 0:  # while the cardinality of set of gray edges is not equal to zero
        u = Q.pop(0)  # Dequeue the first edge
        edges_nei=Graph.edges_neighboring[u] # Neighboring edges

        edges_neigh_n = list(set(edges_nei).intersection(set(result[index]))) # We're only considering the edges that are selected in the territory that's why we're doing the intersection
        for i in edges_neigh_n:  # This is the main loop
            if color[i - 1] == 0:
                color[i - 1] = 1
                d[i - 1] = current_dis + 1
                pred[i - 1] = u
                Q.append(i)  # appending the set of gray nodes

        color[u - 1] = 2

    b = color
    return b




def constraints_wo_cuts(problem, Graph, no_dist, x_v, w_v): #  This function adds all of the constraints for the original districting problem
    # input edges_list, no_nodes,no_edges,no_dist,t
    # sum i e V (xie)=1 for e e E: each edge is assigned to one territory
    expr = [cplex.SparsePair(x_v[(e - 1):Graph.no_edges * Graph.no_nodes:Graph.no_edges],
                             [1 for i in range(Graph.no_nodes)]) for e in Graph.edges_list]
    sens = ["E"] * len(expr)
    rhs = [1] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)

    # sum i e V w_i=p
    sum_x = w_v
    coeff = [1 for i in range(Graph.no_nodes)]
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["E"],
                                   rhs=[no_dist])


    sum_x = []
    coeff = []
    # x_i,(j,k)<=w_i
    expr = [cplex.SparsePair([x_v[index], w_v[i - 1]], [1, -1]) for index, i in enumerate(Graph.node_i)]
    sens = ["L"] * len(expr)
    rhs = [0] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)


    return problem


def SP_contiguity_const(Graph,x_v,problem): # Function that adds the contiguity constraints: x_i,(j,k)<= x_i,(l,m) where (l,m) \in SP_i,(j,k) and (l,m) \in Cutset({(j,k)}) \forall i \in V, \forall (j,k) \in E
    expr = [cplex.SparsePair([x_v[index], x_v[(Graph.node_i[index] - 1) * Graph.no_edges + (
                eval(Graph.nodetoedge_path[Graph.node_i[index]][Graph.edge_e[index]])[-1] - 1)]], [1, -1]) for index, i
            in enumerate(Graph.node_i) if
            len(eval(Graph.nodetoedge_path[Graph.node_i[index]][Graph.edge_e[index]])) > 0]
    sens = ["L"] * len(expr)
    rhs = [0] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    return problem



class Contiguity_Lazy_Callback():
    def __init__(self,x_v,w_v,Graph):
        self.x_v=x_v
        self.w_v=w_v
        self.Graph=Graph
    def lazy_contiguity(self,context):
        x_v=self.x_v
        w_v=self.w_v
        Graph=self.Graph
        result = []
        # print('candidate_solution')
        # print(context.get_candidate_point(w_v))
        center_node1 = [i for i, val in enumerate(context.get_candidate_point(w_v)) if val > 0.01]
        center_node = [i + 1 for i in center_node1]
        for o in center_node:
            result.append([Graph.edge_e[i] for i, val in enumerate(context.get_candidate_point(x_v)) if
                           val > 0.01 and Graph.node_i[i] == o])
        index = 0
        connected_comp = []
        cut_set_for_connected_comp = []
        for o in center_node:  # This loop is to detect whether there are disconnected districts or not
            explored_edges_s1 = []  # edges that are found using BFS
            R = []  # Set of Edges that need to be explored using BFS
            R.extend(result[index])
            if len(R)>0:
                l = R[0]  # the source edge from which we will start exploring
                b = BFS(Graph, index, result, l)
                explored_edges_s1.extend([i + 1 for i, val in enumerate(b) if val == 2])
                explored_edges_s = list(set(explored_edges_s1))
                unexplored_edges = list(
                    set(R).difference(set(explored_edges_s)))  # list of unexplored edges within a district
                connected_comp = []
                cut_set_for_connected_comp = []
                explored_edges_s1 = []  # list of explored edges for every district to keep track of all connected components
                while len(unexplored_edges) > 0:  # This while loop to find all of the different connected components
                    explored_edges_s1.extend([i + 1 for i, val in enumerate(b) if val == 2])
                    explored_edges_s = list(set(explored_edges_s1))
                    unexplored_edges = list(
                        set(R).difference(set(explored_edges_s)))  # list of unexplored edges within a district
                    # Find the connected component
                    # Find the neighboring edges to the connected component (excluding edges in the connected component)
                    # Add the needed cuts
                    connect_edges = [j for i, j in enumerate(R) if b[j - 1] == 2]  # Connected Component (Set Sk)
                    connected_comp.append(connect_edges)
                    connect_edges_neigh_nested = [Graph.edges_neighboring[i] for i in connect_edges]
                    connect_edges_neighboring1 = [j for sublist in connect_edges_neigh_nested for j in sublist]
                    connect_edges_neighboring = set(connect_edges_neighboring1)  # Removing duplicated edges
                    connect_edges_neighboring_n = list(connect_edges_neighboring.difference(
                        set(connect_edges)))  # Neighboring edges to connected component excluding edges in the connected component
                    cut_set_for_connected_comp.append(connect_edges_neighboring_n)
                    if len(unexplored_edges) > 0:  # finding the next connected component
                        l = unexplored_edges[0]
                        b = BFS(Graph, index, result, l)
                index = index + 1
            connected_comp_combin = list(combinations(connected_comp, 2))
            cut_set_for_connected_comb = list(combinations(cut_set_for_connected_comp, 2))

            for combo_index, combo_val in enumerate(connected_comp_combin):
                cut_set_1 = cut_set_for_connected_comb[combo_index][0]
                cut_set_2 = cut_set_for_connected_comb[combo_index][1]
                for ind, val in enumerate(combo_val[1]):
                    s = (- len(combo_val[0]))  # -|S|
                    sum_x = []
                    coeff = []
                    indicies_1 = [(o - 1) * Graph.no_edges + (q - 1) for q in combo_val[0]]
                    x_variables = [x_v[i] for i in indicies_1]
                    sum_x.extend(x_variables)
                    coeff1 = [-1 for r in range(len(indicies_1))]
                    coeff.extend(coeff1)
                    indicies_2 = [(o - 1) * Graph.no_edges + (q - 1) for q in cut_set_1]
                    x_variables_1 = [x_v[i] for i in indicies_2]
                    coeff2 = [1 for r in range(len(indicies_2))]
                    coeff.extend(coeff2)
                    sum_x.extend(x_variables_1)
                    edge_j_k = [val]
                    indicies_3 = [(o - 1) * Graph.no_edges + (q - 1) for q in edge_j_k]
                    x_variables_3 = [x_v[i] for i in indicies_3]
                    coeff3 = [-1]
                    sum_x.extend(x_variables_3)
                    coeff.extend(coeff3)
                    context.reject_candidate(constraints=[cplex.SparsePair(sum_x, coeff), ], senses="G", rhs=[s, ])
                for ind, val in enumerate(combo_val[0]):
                    s = (- len(combo_val[1]))  # -|S|
                    sum_x = []
                    coeff = []
                    indicies_1 = [(o - 1) * Graph.no_edges + (q - 1) for q in combo_val[1]]
                    x_variables = [x_v[i] for i in indicies_1]
                    sum_x.extend(x_variables)
                    coeff1 = [-1 for r in range(len(indicies_1))]
                    coeff.extend(coeff1)
                    indicies_2 = [(o - 1) * Graph.no_edges + (q - 1) for q in cut_set_2]
                    x_variables_1 = [x_v[i] for i in indicies_2]
                    coeff2 = [1 for r in range(len(indicies_2))]
                    coeff.extend(coeff2)
                    sum_x.extend(x_variables_1)
                    edge_j_k = [val]
                    indicies_3 = [(o - 1) * Graph.no_edges + (q - 1) for q in edge_j_k]
                    x_variables_3 = [x_v[i] for i in indicies_3]
                    coeff3 = [-1]
                    sum_x.extend(x_variables_3)
                    coeff.extend(coeff3)
                    context.reject_candidate(constraints=[cplex.SparsePair(sum_x, coeff), ], senses="G", rhs=[s, ])

    def invoke(self, context):
        try:
            if context.in_candidate():
                self.lazy_contiguity(context)
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


class Model(): # This is the model where we add the variables and the constraints
    def __init__(self,Graph, x_v, w_v,num):
        self.x_v=x_v
        self.w_v=w_v
        self.Graph=Graph
        c_org = cplex.Cplex()
        self.no_threads = 'unlimited'
        # c_org.parameters.threads.set(self.no_threads)
        c_org.parameters.mip.strategy.file.set(0)


        # Setting the objective function to be Minmization
        c_org.objective.set_sense(c_org.objective.sense.minimize)
        # Declare decision variables (first argument is decision variables names, second argument is type of decision variables,
        # third argument is objective function coefficients)
        nodetoedge_net_dist_list = Graph.nodetoedge_net_dist.stack().tolist()  # converting the dataframe into a list
        c_org.variables.add(names=x_v, types=["B"] * len(x_v), obj=nodetoedge_net_dist_list)
        c_org.variables.add(names=w_v, types=["B"] * len(w_v))
        c_main=constraints_wo_cuts(c_org, Graph, num, x_v, w_v)
        self.c_main=c_main

    def branch_bound_cut(self): # This is the method that has the branch and bound and cut
        x_v = self.x_v
        w_v = self.w_v
        Graph = self.Graph
        c_1=self.c_main
        t0 = time.time()
        d=c_1
        d.parameters.clocktype.set(2)
        d.parameters.timelimit.set(43200)
        contiguity_callback = Contiguity_Lazy_Callback(x_v, w_v, Graph)
        contextmask = cplex.callbacks.Context.id.candidate
        d.set_callback(contiguity_callback, contextmask)
        d.solve()
        l_1 = round(time.time() - t0, 2)
        print('Time to Solve Using Branch and Cut')
        print(l_1)
        sol_1 = [x_v[i] for i, j in enumerate(d.solution.get_values(x_v)) if j > 0.01]
        sol_2 = [w_v[i] for i, j in enumerate(d.solution.get_values(w_v)) if j > 0.01]
        obj = d.solution.get_objective_value()
        print("gap tolerance = ", d.parameters.mip.tolerances.mipgap.get())
        print(sol_1)
        print(sol_2)
        print(obj)

        return obj,l_1,d

    def solve_poly_cont(self):
        Graph = self.Graph
        x_v = self.x_v
        w_v = self.w_v
        c_2=self.c_main
        t0 = time.time()
        c=SP_contiguity_const(Graph, x_v, c_2)
        # c_logic=logic_cuts(Graph, c, w_v, x_v)
        c.parameters.clocktype.set(2)
        c.parameters.timelimit.set(43200)
        c.solve()
        l_1 = round(time.time() - t0, 2)
        sol_1 = [x_v[i] for i, j in enumerate(c.solution.get_values(x_v)) if j > 0.01]
        sol_2 = [w_v[i] for i, j in enumerate(c.solution.get_values(w_v)) if j > 0.01]
        obj = c.solution.get_objective_value()
        print('Solution')
        print(sol_1)
        print(sol_2)
        print('Objective Function')
        print(obj)
        return obj,l_1,c


def execute_task(task):
    print(f"Instance Name {task[0]}")
    print(f"No. of Districts {task[1]}")
    prob=task[0]
    num=task[1]
    graph = Graph(prob)
    nodes_list = graph.nodes_list
    edges_list = graph.edges_list
    no_nodes = graph.no_nodes
    no_edges = graph.no_edges
    node_i = graph.node_i
    edge_e = graph.edge_e
    # Creating a list of variable names (x_i,(j,k)): binary variable whether the edge (j,k) is assigned to district i
    x_v = []
    for i in range(len(node_i)):
        x = 'x' + str(node_i[i]) + '_' + str(edge_e[i])
        x_v.append(x)

    # Creating a list of variable names (wi): binary variable whether the node i is the center or not
    w_v = []
    for i in range(len(nodes_list)):
        w = 'w' + str(nodes_list[i])
        w_v.append(w)

    distr_v = list(range(1, num + 1))  # Districts vector
    model_bbb = Model(graph, x_v, w_v, num)
    no_threads = model_bbb.no_threads
    with open('Results/Cut_Empty/'+str(no_threads)+'_threads/ECPM_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) + '_prob_' + str(prob) + '.csv',
              'w') as newFile:
        print('Directory')
        print('Results/Cut_Empty/'+str(no_threads)+'_threads/ECPM_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) + '_prob_' + str(prob) + '.csv')
        newFileWriter = csv.writer(newFile, lineterminator='\n')
        newFileWriter.writerow(['num_threads'])
        newFileWriter.writerow([no_threads])

        # try:
        #     newFileWriter.writerow(['Computation Time_for_Branch_Cut', 'Objective Function'])
        #     obj, l_1, d = model_bbb.branch_bound_cut()
        #     lower_bound = d.solution.MIP.get_best_objective()
        #     relative_gap = d.solution.MIP.get_mip_relative_gap()
        #     soln_status = d.solution.get_status_string()
        #     newFileWriter.writerow([l_1, round(obj, 2)])
        #     newFileWriter.writerow(['Lower_Bound', 'Relative Gap', 'Solution Status'])
        #     newFileWriter.writerow([round(lower_bound, 2), relative_gap, soln_status])
        # except cplex.exceptions.CplexError as e:
        #     print("CPLEX Error", e)
        #     newFileWriter.writerow(['CPLEX Error'])
        #     newFileWriter.writerow([e])

        print('Start of SPC Formulation')
        try:
            model_sp_cont = Model(graph, x_v, w_v, num)
            obj, l_1, c = model_sp_cont.solve_poly_cont()
            lower_bound_w_cuts = c.solution.MIP.get_best_objective()
            relative_gap_w_cuts = c.solution.MIP.get_mip_relative_gap()
            soln_status_w_cuts = c.solution.get_status_string()
            newFileWriter.writerow(['Computation Time_Empty_SP_Cont', 'Objective Function'])
            newFileWriter.writerow([l_1, round(obj, 2)])
            newFileWriter.writerow(['Lower_Bound','Relative Gap','Solution Status'])
            newFileWriter.writerow([round(lower_bound_w_cuts, 2),relative_gap_w_cuts,soln_status_w_cuts])
        except cplex.exceptions.CplexError as e:
            print("CPLEX Error", e)
            newFileWriter.writerow(['CPLEX Error'])
            newFileWriter.writerow([e])



# For testing purposes
# task=['EBD_SP_Cut_Empty','CARP_N17_g_graph.dat',50,1]
# execute_task(task)

if __name__ == "__main__":
    # List of tasks: [Model Name, Problem File Name, Number of Districts, Balance Tolerance]
    task_list = [
        # [ "CARP_F6_p_graph.dat", 2],
        # ["CARP_F6_p_graph.dat", 10],
        # ["CARP_O12_g_graph.dat", 2],
        # ["CARP_O12_g_graph.dat", 10],
        # ["CARP_N16_g_graph.dat", 2],
        # ["CARP_N16_g_graph.dat", 10],
        ["CARP_S9_g_graph.dat", 2],
        ["CARP_S9_g_graph.dat", 10]
        # Add more tasks here as needed
    ]

    for task in task_list:
        try:
            execute_task(task)
        except Exception as e:
            print(f"Error while executing task {task}: {e}")
