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
        self.demand=np.loadtxt(self.file, skiprows=9, max_rows=self.no_edges1, usecols=5, dtype=float)
        # This block calculates the length of each edge
        index = 0
        self.euc_dist = collections.defaultdict(dict)  # dictionary to store the Euclidean Distance of each edge, since it is a symmetric matrix we add the below two lines
        for ind, j in enumerate(self.from_node_i):
            self.euc_dist[self.from_node_i[ind] + 1][self.to_node_j[index] + 1] = self.edge_lengths[index]
            self.euc_dist[self.to_node_j[ind] + 1][self.from_node_i[index] + 1] = self.edge_lengths[index]
            index += 1

        # # creating a vector of index for every pair of node (i) and edge (e)
        self.node_i = []
        self.node_j = []
        for i in self.nodes_list:
            for j in self.nodes_list:
                self.node_i.append(i)
                self.node_j.append(j)

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



        # # Creating a list of the degree for every node
        self.degree = []
        for i in self.nodes_list:
            self.degree.append(len(self.edges[i]))


        self.nodetonode_net_dist = pd.read_csv("nodetonode_distance_" + str(prob) + ".csv", header=0, index_col=0)
        self.nodetonode_path = pd.read_csv("nodetonode_path_" + str(prob) + ".csv", index_col=0, header=0)
        self.nodetonode_path.columns = self.nodes_list
        self.nodetonode_path = self.nodetonode_path.transpose()
        self.nodetonode_path_dict = self.nodetonode_path.to_dict()


## End of Data

# Breadth First Function
def BFS(Graph,index, result, l):
    # defining the arrays that will be in the algorithm
    pred = []  # predecessor vector
    color = []  # color vector, 0 means white, 1 means gray ,2 means black
    d = []  # distance vector
    Q = []  # set of gray vectors
    s = l  # the source edge index in the edge vector
    for node in Graph.nodes_list:
        color.append(int(0))  # having all the colors of edges to be white
        d.append(int(0))
        pred.append(int(0))

    color[s - 1] = 1
    Q.append(s)
    current_dis = 0

    while len(Q) != 0:  # while the cardinality of set of gray edges is not equal to zero
        u = Q.pop(0)  # Dequeue the first edge
        nodes_nei=Graph.adj[u] # Neighboring nodes

        nodes_neigh_n = list(set(nodes_nei).intersection(set(result[index]))) # We're only considering the edges that are selected in the territory that's why we're doing the intersection
        for i in nodes_neigh_n:  # This is the main loop
            if color[i - 1] == 0:
                color[i - 1] = 1
                d[i - 1] = current_dis + 1
                pred[i - 1] = u
                Q.append(i)  # appending the set of gray nodes

        color[u - 1] = 2

    b = color
    return b



def constraints_wo_cuts(problem, Graph, no_dist,tol, x_v): #  This function adds all of the constraints for the original districting problem
    # input edges_list, no_nodes,no_edges,no_dist,t
    # sum i e V (xij)=1 for j \in V: each node is assigned to one territory
    expr = [cplex.SparsePair(x_v[(j - 1):Graph.no_nodes * Graph.no_nodes:Graph.no_nodes],
                             [1 for i in range(Graph.no_nodes)]) for j in Graph.nodes_list]
    sens = ["E"] * len(expr)
    rhs = [1] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)

    # sum i e V x_ii=p
    sum_x = [j for i,j in enumerate(x_v) if Graph.node_i[i]==Graph.node_j[i]]
    coeff = [1 for i in range(Graph.no_nodes)]
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["E"],
                                   rhs=[no_dist])

    sum_x = []
    coeff = []
    # Balancing Constraints
    # sum j e V le xij <= |V|/p * (1+tau) xii for i e V
    rhs_1 = (Graph.no_nodes/ no_dist) * (1 + tol)

    expr = [cplex.SparsePair([x_v[i - 1] if Graph.node_i[i]] + x_v[(i - 1) * Graph.no_nodes:i * Graph.no_nodes],
                             [-rhs_1] + list(Graph.demand_edge_index.values())) for i in Graph.nodes_list]
    sens = ["L"] * len(expr)
    rhs = [0] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    # sum e e E le xie >= sum(le)/p * (1-tau) wi for i e V
    rhs_2 = (sum(Graph.demand_edge_index.values()) / no_dist) * (1 - tol)
    expr = [cplex.SparsePair([w_v[i - 1]] + x_v[(i - 1) * Graph.no_edges:i * Graph.no_edges],
                             [-rhs_2] + list(Graph.demand_edge_index.values())) for i in Graph.nodes_list]
    rhs = [0] * len(expr)
    sens = ["G"] * len(expr)
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
        self.Graph=Graph
    def lazy_contiguity(self,context):
        x_v=self.x_v
        Graph=self.Graph
        result = []
        # print('candidate_solution')
        # print(context.get_candidate_point(w_v))
        center_node1 = [i for i, val in enumerate(context.get_candidate_point(x_v)) if Graph.node_i[i] == Graph.node_j[i] and val > 0.01]
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
    def __init__(self,Graph, x_v, num,tol):
        self.x_v=x_v
        self.Graph=Graph
        c_org = cplex.Cplex()
        self.no_threads = 'unlimited'
        # c_org.parameters.threads.set(self.no_threads)
        c_org.parameters.mip.strategy.file.set(3)
        c_org.parameters.workdir.set('/scratch/zekassem/nodefile')

        # Setting the objective function to be Minmization
        c_org.objective.set_sense(c_org.objective.sense.minimize)
        # Declare decision variables (first argument is decision variables names, second argument is type of decision variables,
        # third argument is objective function coefficients)
        nodetonode_net_dist_list = Graph.nodetonode_net_dist.stack().tolist()  # converting the dataframe into a list
        c_org.variables.add(names=x_v, types=["B"] * len(x_v), obj=nodetonode_net_dist_list)
        c_main=constraints_wo_cuts(c_org, Graph, num,tol, x_v)
        self.c_main=c_main

    def branch_bound_cut(self): # This is the method that has the branch and bound and cut
        x_v = self.x_v
        Graph = self.Graph
        c_1=self.c_main
        t0 = time.time()
        d=c_1
        d.parameters.clocktype.set(2)
        d.parameters.timelimit.set(43200)
        contiguity_callback = Contiguity_Lazy_Callback(x_v, Graph)
        contextmask = cplex.callbacks.Context.id.candidate
        d.set_callback(contiguity_callback, contextmask)
        d.solve()
        l_1 = round(time.time() - t0, 2)
        print('Time to Solve Using Branch and Cut')
        print(l_1)
        sol_1 = [x_v[i] for i, j in enumerate(d.solution.get_values(x_v)) if j > 0.01]
        obj = d.solution.get_objective_value()
        print("gap tolerance = ", d.parameters.mip.tolerances.mipgap.get())
        print(sol_1)
        print(obj)

        return obj,l_1,d

    def solve_poly_cont(self):
        Graph = self.Graph
        x_v = self.x_v
        c_2=self.c_main
        t0 = time.time()
        c=SP_contiguity_const(Graph, x_v, c_2)
        # c_logic=logic_cuts(Graph, c, w_v, x_v)
        c.parameters.clocktype.set(2)
        c.parameters.timelimit.set(43200)
        c.solve()
        l_1 = round(time.time() - t0, 2)
        sol_1 = [x_v[i] for i, j in enumerate(c.solution.get_values(x_v)) if j > 0.01]
        obj = c.solution.get_objective_value()
        print('Solution')
        print(sol_1)
        print('Objective Function')
        print(obj)
        return obj,l_1,c

def execute_task(task):
    print(f"Model Name {task[0]}")
    print(f"Instance Name {task[1]}")
    print(f"No. of Districts {task[2]}")
    print(f"Balance Tolerance {task[3]}")
    prob=task[1]
    num=task[2]
    tol=task[3]
    graph = Graph(prob)
    nodes_list = graph.nodes_list
    edges_list = graph.edges_list
    no_nodes = graph.no_nodes
    no_edges = graph.no_edges
    node_i = graph.node_i
    node_j = graph.node_j
    # Creating a list of variable names (x_i,(j,k)): binary variable whether the edge (j,k) is assigned to district i
    x_v = []
    for i in range(len(node_i)):
        x = 'x' + str(node_i[i]) + '_' + str(node_j[i])
        x_v.append(x)


    distr_v = list(range(1, num + 1))  # Districts vector
    model_bbb = Model(graph, x_v, num, tol)
    no_threads = model_bbb.no_threads
    with open('Results/Cut_Empty/'+str(no_threads)+'_threads/EBD_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) +'_tol_'+str(tol)+ '_prob_' + str(prob) + '.csv',
              'w') as newFile:
        print('Directory')
        print('Results/Cut_Empty/'+str(no_threads)+'_threads/EBD_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) +'_tol_'+str(tol)+ '_prob_' + str(prob) + '.csv')
        newFileWriter = csv.writer(newFile, lineterminator='\n')
        newFileWriter.writerow(['num_threads'])
        newFileWriter.writerow([no_threads])

        try:
            newFileWriter.writerow(['Computation Time_for_Branch_Cut', 'Objective Function'])
            obj, l_1, d = model_bbb.branch_bound_cut()
            lower_bound = d.solution.MIP.get_best_objective()
            relative_gap = d.solution.MIP.get_mip_relative_gap()
            soln_status = d.solution.get_status_string()
            newFileWriter.writerow([l_1, round(obj, 2)])
            newFileWriter.writerow(['Lower_Bound', 'Relative Gap', 'Solution Status'])
            newFileWriter.writerow([round(lower_bound, 2), relative_gap, soln_status])
        except cplex.exceptions.CplexError as e:
            print("CPLEX Error", e)
            newFileWriter.writerow(['CPLEX Error'])
            newFileWriter.writerow([e])

        # print('Start of SPC Formulation')
        # try:
        #     model_sp_cont = Model(graph, x_v, w_v, num, tol)
        #     obj, l_1, c = model_sp_cont.solve_poly_cont()
        #     lower_bound_w_cuts = c.solution.MIP.get_best_objective()
        #     relative_gap_w_cuts = c.solution.MIP.get_mip_relative_gap()
        #     soln_status_w_cuts = c.solution.get_status_string()
        #     newFileWriter.writerow(['Computation Time_Empty_SP_Cont', 'Objective Function'])
        #     newFileWriter.writerow([l_1, round(obj, 2)])
        #     newFileWriter.writerow(['Lower_Bound','Relative Gap','Solution Status'])
        #     newFileWriter.writerow([round(lower_bound_w_cuts, 2),relative_gap_w_cuts,soln_status_w_cuts])
        # except cplex.exceptions.CplexError as e:
        #     print("CPLEX Error", e)
        #     newFileWriter.writerow(['CPLEX Error'])
        #     newFileWriter.writerow([e])



# For testing purposes
# task=['EBD_SP_Cut_Empty','CARP_N17_g_graph.dat',2,1]
# execute_task(task)