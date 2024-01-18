import time
import numpy as np
import collections
import csv
import pandas as pd
import cplex
from itertools import combinations

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
        self.euc_dist = collections.defaultdict(
            dict)  # dictionary to store the Euclidean Distance of each edge, since it is a symmetric matrix we add the below two lines
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


def constraints_wo_cuts(problem, Graph, no_dist,x_v, w_v): #  This function adds all of the constraints for the original districting problem
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



class Model(): # This is the model where we add the variables and the constraints
    def __init__(self,Graph, x_v, w_v,num):
        self.x_v=x_v
        self.w_v=w_v
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
        nodetoedge_net_dist_list = Graph.nodetoedge_net_dist.stack().tolist()  # converting the dataframe into a list
        c_org.variables.add(names=x_v, types=["B"] * len(x_v),obj=nodetoedge_net_dist_list)
        c_org.variables.add(names=w_v, types=["B"] * len(w_v))
        c_main=constraints_wo_cuts(c_org, Graph, num, x_v, w_v)
        self.c_main=c_main

    def solve_EPM(self):
        Graph = self.Graph
        x_v = self.x_v
        w_v = self.w_v
        c=self.c_main
        t0 = time.time()
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
    # print(f"Balance Tolerance {task[3]}")
    prob=task[0]
    num=task[1]
    # tol=task[3]
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
    with open('Results/Cut_Empty/'+str(no_threads)+'_threads/EPM_no_dist_' + str(num) +'_prob_' + str(prob) + '.csv',
              'w') as newFile:
        print('Directory')
        print('Results/Cut_Empty/'+str(no_threads)+'_threads/EPM_no_dist_' + str(num) +'_prob_' + str(prob) + '.csv')
        newFileWriter = csv.writer(newFile, lineterminator='\n')
        newFileWriter.writerow(['num_threads'])
        newFileWriter.writerow([no_threads])

        print('Start of SPC Formulation')
        try:
            model_EPM = Model(graph, x_v, w_v, num)
            obj, l_1, c = model_EPM.solve_EPM()
            lower_bound_w_cuts = c.solution.MIP.get_best_objective()
            relative_gap_w_cuts = c.solution.MIP.get_mip_relative_gap()
            soln_status_w_cuts = c.solution.get_status_string()
            newFileWriter.writerow(['Computation Time_EPM', 'Objective Function'])
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