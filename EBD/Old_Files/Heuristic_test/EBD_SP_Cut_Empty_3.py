import time
import numpy as np
import collections
import csv
import pandas as pd
import cplex
import itertools
import copy

# Loading Graph
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
# Input parameters to the districting problem
no_dist_list=[2] # the number of districts
tol_list=[0.01] #Balance Tolerance

# Breadth First Function
def BFS(Graph,district_edges):
    # defining the arrays that will be in the algorithm
    pred = []  # predecessor vector
    color = []  # color vector, 0 means white, 1 means gray ,2 means black
    d = []  # distance vector
    Q = []  # set of gray vectors
    s = district_edges[0]  # the source edge index in the edge vector
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

        edges_neigh_n = list(set(edges_nei).intersection(set(district_edges))) # We're only considering the edges that are selected in the territory that's why we're doing the intersection
        for i in edges_neigh_n:  # This is the main loop
            if color[i - 1] == 0:
                color[i - 1] = 1
                d[i - 1] = current_dis + 1
                pred[i - 1] = u
                Q.append(i)  # appending the set of gray nodes

        color[u - 1] = 2

    b = color
    return b


def Dijkstras_Algo(Graph,center_node_index,edges_list,nodes_list): # Lazy Implementation of Dijkstra's Algorithm to find the shortest path from the center node to each edge
    network_dist=collections.defaultdict(dict) # nested dictionary to store the network distance from and to each node
    pred=collections.defaultdict(dict) # nested dictionary to store the predecessor of each node on the shortest path
    for index,source in enumerate(nodes_list):
        unvisited_nodes=copy.copy(nodes_list) #list of unvisited_nodes
        for i,j in enumerate(nodes_list):
            network_dist[source][j]=float('inf')  # Adding the infinity label to all nodes
            pred[source][j]=0 #Adding 0 predecessor to all nodes
        network_dist[source][source]=0
        while unvisited_nodes:
            dist_unvisited={k:network_dist[source][k] for k in unvisited_nodes} #getting the distances of unvisited nodes
            node_min=min(dist_unvisited,key=dist_unvisited.get) #getting the node of min. distance within the unvisited nodes
            unvisited_nodes.remove(node_min) #remove the unvisited node with minimum distance
            neigh_nodes=list(set(Graph.adj[node_min])&set(nodes_list)) # list of neighboring nodes
            edges_list_node_index=[Graph.nodes_corr[b] for b in edges_list] # list of edges
            for l,k in enumerate(neigh_nodes):
                if [k,node_min] in edges_list_node_index or [node_min,k] in edges_list_node_index: #Checking that the edge exists in the district
                    new_dist=network_dist[source][node_min]+Graph.euc_dist[node_min][k]
                    if new_dist<network_dist[source][k]:
                        network_dist[source][k]=new_dist
                        pred[source][k]=node_min

    # This for loop for printing the shortest path between every two nodes
    path_nodes = collections.defaultdict(dict)  # Nested Dictionary that contains the shortest path in terms of nodes
    for node_index, node in enumerate(nodes_list):
        for target_index, target_node in enumerate(nodes_list):
            path_list = []
            u = target_node
            path_list.insert(0, u)
            while pred[node][u] != 0:  # Backtracking the Shortest Path
                path_list.insert(0, pred[node][u])
                u = pred[node][u]
            path_nodes[node][target_node] = path_list

    path_edges = collections.defaultdict(
        dict)  # Nested Dictionary that has the shortest path between each two nodes based on edges defined [i,j]

    # This block creates the shortest path between each two nodes based on edges defined [i,j]
    for i in nodes_list:
        for j in nodes_list:
            if i == j:
                path_edges[i][j] = []
            if i != j:
                path_current = path_nodes[i][j]
                path_edges[i][j] = [[path_current[k], path_current[k + 1]] for k, l in enumerate(path_current) if
                                    k + 1 < len(path_current)]

    # This block creates the shortest path between each two nodes based on edge number
    for i in nodes_list:
        for j in nodes_list:
            if i == j:
                path_edges[i][j] = []
            if i != j:
                path_current = path_edges[i][j]
                path_new = []
                for k, l in enumerate(path_current):
                    path_new.extend([key for key, value in Graph.nodes_corr.items() if
                                     sorted(value) == sorted(path_current[k])])
                path_edges[i][j] = path_new

    # getting node to edge distance after getting the shortest path between every two nodes
    nodetoedge_net_dist = collections.defaultdict(dict) # Create a nested dictionary to store the distance between each node and edge within the district
    nodetoedge_path=collections.defaultdict(dict) # Create a nested dictionary to store the path between each node and edge within the district

    # creating the min node-to-edge distances nested dictionary, node-to-edge path nested dictionary:

    for e in edges_list:
        nodes_corr = Graph.nodes_corr[e]
        network_dist_1 = network_dist[center_node_index][nodes_corr[0]]
        network_dist_2 = network_dist[center_node_index][nodes_corr[1]]
        nodetoedge_net_dist[center_node_index][e] = min(network_dist_1,network_dist_2)  # calculating the minimum of the two distances
        if network_dist_1<network_dist_2:
            nodetoedge_path[center_node_index][e]=path_edges[center_node_index][nodes_corr[0]]
        else:
            nodetoedge_path[center_node_index][e]=path_edges[center_node_index][nodes_corr[1]]

    return nodetoedge_net_dist,nodetoedge_path

def constraints_wo_cuts(problem, Graph, no_dist,tol, x_v, w_v): #  This function adds all of the constraints for the original districting problem
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
    # Balancing Constraints
    # sum e e E le xie <= sum(le)/p * (1+tau) wi for i e V
    rhs_1 = (sum(Graph.euc_dist_edge_index.values()) / no_dist) * (1 + tol)

    expr = [cplex.SparsePair([w_v[i - 1]] + x_v[(i - 1) * Graph.no_edges:i * Graph.no_edges],
                             [-rhs_1] + list(Graph.euc_dist_edge_index.values())) for i in Graph.nodes_list]
    sens = ["L"] * len(expr)
    rhs = [0] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    # sum e e E le xie >= sum(le)/p * (1-tau) wi for i e V
    rhs_2 = (sum(Graph.euc_dist_edge_index.values()) / no_dist) * (1 - tol)
    expr = [cplex.SparsePair([w_v[i - 1]] + x_v[(i - 1) * Graph.no_edges:i * Graph.no_edges],
                             [-rhs_2] + list(Graph.euc_dist_edge_index.values())) for i in Graph.nodes_list]
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

class Model(): # This is the model where we add the variables and the constraints
    def __init__(self,Graph, x_v, w_v,num,tol):
        self.x_v=x_v
        self.w_v=w_v
        self.Graph=Graph
        c_org = cplex.Cplex()
        self.no_threads = 'unlimited'
        # c_org.parameters.threads.set(self.no_threads)
        #c_org.parameters.mip.strategy.file.set(2)
        # Setting the objective function to be Minmization
        c_org.objective.set_sense(c_org.objective.sense.minimize)
        # Declare decision variables (first argument is decision variables names, second argument is type of decision variables,
        # third argument is objective function coefficients)
        nodetoedge_net_dist_list = Graph.nodetoedge_net_dist.stack().tolist()  # converting the dataframe into a list
        c_org.variables.add(names=x_v, types=["B"] * len(x_v), obj=nodetoedge_net_dist_list)
        c_org.variables.add(names=w_v, types=["B"] * len(w_v))
        c_main=constraints_wo_cuts(c_org, Graph, num,tol, x_v, w_v)
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
        d.solve()
        l_1 = round(time.time() - t0, 2)
        print('time to solve before BBB')
        print(l_1)
        t1 = time.time()
        result=[]
        center_node1 = [i for i, val in enumerate(d.solution.get_values(w_v)) if val > 0.01]
        center_node = [i + 1 for i in center_node1]
        for o in center_node:
            result.append([Graph.edge_e[i] for i,val in enumerate(d.solution.get_values(x_v)) if val>0.01 and Graph.node_i[i]==o])
        a = 1
        num_iterations=0 # counter for the number of iterations
        num_cuts=0 # counter for the number of
        while a > 0:
            num_iterations+=1
            C = []
            index = 0
            for o in center_node:  # This loop is to detect whether there are disconnected districts or not
                print('Hi')
                explored_edges_s1 = []  # edges that are found using BFS
                R = []  # Set of Edges that need to be explored using BFS
                R.extend(result[index])
                l = R[0]  # the source edge from which we will start exploring
                b = BFS(Graph, result[index])
                explored_edges_s1.extend([i + 1 for i, val in enumerate(b) if val == 2])
                explored_edges_s = list(set(explored_edges_s1))
                unexplored_edges = list(
                    set(R).difference(set(explored_edges_s)))  # list of unexplored edges within a district
                if len(unexplored_edges) > 0:
                    C.append(0)
                else:
                    C.append(1)
                explored_edges_s1 = []  # list of explored edges for every district to keep track of all connected components
                while len(
                        unexplored_edges) > 0:  # This while loop to find all of the different connected components and add all of the needed cuts
                    explored_edges_s1.extend([i + 1 for i, val in enumerate(b) if val == 2])
                    explored_edges_s = list(set(explored_edges_s1))
                    unexplored_edges = list(
                        set(R).difference(set(explored_edges_s)))  # list of unexplored edges within a district
                    # Find the connected component
                    # Find the neighboring edges to the connected component (excluding edges in the connected component)
                    # Add the needed cuts
                    connect_edges = [j for i, j in enumerate(R) if b[j - 1] == 2]  # Connected Component (Set Sk)
                    connect_edges_neigh_nested = [Graph.edges_neighboring[i] for i in connect_edges]
                    connect_edges_neighboring1 = [j for sublist in connect_edges_neigh_nested for j in sublist]
                    connect_edges_neighboring = set(connect_edges_neighboring1)  # Removing duplicated edges
                    connect_edges_neighboring_n = list(connect_edges_neighboring.difference(
                        set(
                            connect_edges)))  # Neighboring edges to connected component excluding edges in the connected component
                    s = (1 - len(connect_edges))  # 1-|S|
                    sum_x = []
                    coeff = []
                    indicies_1 = [(o - 1) * Graph.no_edges + (q - 1) for q in connect_edges]
                    x_variables = [x_v[i] for i in indicies_1]
                    sum_x.extend(x_variables)
                    coeff1 = [-1 for r in range(len(indicies_1))]
                    coeff.extend(coeff1)
                    indicies_2 = [(o - 1) * Graph.no_edges + (q - 1) for q in connect_edges_neighboring_n]
                    x_variables_1 = [x_v[i] for i in indicies_2]
                    coeff2 = [1 for r in range(len(indicies_2))]
                    coeff.extend(coeff2)
                    sum_x.extend(x_variables_1)
                    if len(sum_x)>0:
                        num_cuts+=1
                    d.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["G"], rhs=[s])
                    if len(unexplored_edges) > 0:  # finding the next connected component
                        l = unexplored_edges[0]
                        b = BFS(Graph, result[index])
                index = index + 1
            if sum(C) < len(center_node):
                a = 1
                d.solve()
                center_node1 = [i for i, val in enumerate(d.solution.get_values(w_v)) if val > 0.01]
                center_node = [i + 1 for i in center_node1]
                result = []
                for o in center_node:
                    result.append(
                        [Graph.edge_e[i] for i, val in enumerate(d.solution.get_values(x_v)) if val > 0.01 and Graph.node_i[i] == o])
            else:
                a = 0

            print('num_cuts')
            print(num_cuts)
            print('num_iterations')
            print(num_iterations)

        l_2 = round(time.time() - t1, 2)
        sol_1 = [x_v[i] for i, j in enumerate(d.solution.get_values(x_v)) if j > 0.01]
        sol_2 = [w_v[i] for i, j in enumerate(d.solution.get_values(w_v)) if j > 0.01]
        obj = d.solution.get_objective_value()
        print("gap tolerance = ", d.parameters.mip.tolerances.mipgap.get())
        center_node = [i + 1 for i, val in enumerate(d.solution.get_values(w_v)) if val > 0.01]
        districts_edges = {}
        for o in center_node:
            districts_edges[o] = [Graph.edge_e[i] for i, val in enumerate(d.solution.get_values(x_v)) if
                                  val > 0.01 and Graph.node_i[i] == o]
        districts_nodes = collections.defaultdict(list)
        for o in center_node:
            districts_nodes[o].extend([Graph.nodes_corr[i] for i in districts_edges[o]])
        for o in center_node:
            districts_nodes[o] = list(set(list(itertools.chain(*districts_nodes[o]))))

        return obj,l_1,l_2,num_iterations, num_cuts,d,center_node,districts_edges,districts_nodes

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
        center_node = [i + 1 for i, val in enumerate(c.solution.get_values(w_v)) if val > 0.01]
        districts_edges = {}
        for o in center_node:
            districts_edges[o] = [Graph.edge_e[i] for i, val in enumerate(c.solution.get_values(x_v)) if
                                  val > 0.01 and Graph.node_i[i] == o]

        districts_nodes = collections.defaultdict(list)
        for o in center_node:
            districts_nodes[o].extend([Graph.nodes_corr[i] for i in districts_edges[o]])
        for o in center_node:
            districts_nodes[o] = list(set(list(itertools.chain(*districts_nodes[o]))))

        return obj,l_1,c,center_node,districts_edges,districts_nodes

def heuristic(Graph,center_node,districts_edges,tol,num,obj):
    lenght_list = [Graph.euc_dist_edge_index[q] for q in Graph.edges_list]
    total_length_b = sum(lenght_list)
    UB = (total_length_b / num) * (1 + tol)
    LB = (total_length_b / num) * (1 - tol)
    # Loop through the districts
    count=0
    for o in center_node:
        # the next four lines finds the cut set of E_o
        connect_edges_neigh_nested = [Graph.edges_neighboring[i] for i in districts_edges[o]]
        connect_edges_neighboring1 = [j for sublist in connect_edges_neigh_nested for j in sublist]
        connect_edges_neighboring = set(connect_edges_neighboring1)  # Removing duplicated edges
        connect_edges_neighboring_n = list(connect_edges_neighboring.difference(
            set(
                districts_edges[o])))  # Neighboring edges to connected component excluding edges in the connected component

        reverse_dict = {value: key for key, values in districts_edges.items() for value in values}
        # This loop goes through all of the edges in the cut set of E_o
        for e in connect_edges_neighboring_n:
            # find the district center
            u=reverse_dict[e]
            #get the distance between d_o,e d_u,e
            d_oe=(Graph.nodetoedge_net_dist.iloc[[o - 1], [e - 1]])
            d_oe_value=d_oe.values.tolist()[0]
            d_ue=Graph.nodetoedge_net_dist.iloc[[u - 1], [e - 1]]
            d_ue_value = d_ue.values.tolist()[0]
            e_length=Graph.euc_dist_edge_index[e]
            district_o_total_demand=sum([Graph.euc_dist_edge_index[l] for l in districts_edges[o]])
            district_u_total_demand=sum([Graph.euc_dist_edge_index[l] for l in districts_edges[u]])
            new_district_o_total_demand=district_o_total_demand+e_length
            new_district_u_total_demand=district_u_total_demand-e_length
            if d_oe_value[0]<d_ue_value[0] and new_district_o_total_demand<=UB and new_district_u_total_demand>=LB:
                count+=1
                districts_edges[o].append(e)
                districts_edges[u].remove(e)
                b_o=BFS(Graph,districts_edges[o])
                b_u=BFS(Graph,districts_edges[u])
                if sum(b_o)==2*len(districts_edges[o]) and sum(b_u)==2*len(districts_edges[u]):
                    obj=obj+d_oe_value[0]-d_ue_value[0]
                else:
                    print('District is disconnected')
        return obj




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
    model_bbb = Model(graph, x_v, w_v, num, tol)
    no_threads = model_bbb.no_threads
    with open('Results/Cut_Empty/'+str(no_threads)+'_threads/EBD_Cut_Set_vs_SP_Contiguity_no_dist_' + str(num) +'_tol_'+str(tol)+ '_prob_' + str(prob) + '.csv',
              'w') as newFile:
        newFileWriter = csv.writer(newFile, lineterminator='\n')
        newFileWriter.writerow(['num_threads'])
        newFileWriter.writerow([no_threads])

        try:
            newFileWriter.writerow(['Computation Time_before_BBB', 'Objective Function'])
            obj, l_1, l_2, num_iterations, num_cuts, d,center_node_1,districts_edges_1,districts_nodes_1 = model_bbb.branch_bound_cut()
            print('Cut Set Obj')
            print(obj)
            lower_bound = d.solution.MIP.get_best_objective()
            relative_gap = d.solution.MIP.get_mip_relative_gap()
            soln_status = d.solution.get_status_string()
            newFileWriter.writerow([l_1, round(obj, 2)])
            newFileWriter.writerow(['Computation Time_for_BBB','num_iterations', 'num_cuts'])
            newFileWriter.writerow([l_2, num_iterations, num_cuts])
            newFileWriter.writerow(['Lower_Bound', 'Relative Gap', 'Solution Status'])
            newFileWriter.writerow([round(lower_bound, 2), relative_gap, soln_status])
            nodetoedge_net_dist_all_1 = {}  # dictionary that has all of the distances
            nodetoedge_net_path_all_1 = {}  # dictionary that has all of the paths
            for o in center_node_1:
                center_node_index = o
                nodetoedge_net_dist, nodetoedge_path = Dijkstras_Algo(graph, center_node_index,
                                                                      districts_edges_1[center_node_index],
                                                                      districts_nodes_1[center_node_index])
                nodetoedge_net_dist_all_1.update(nodetoedge_net_dist)
                nodetoedge_net_path_all_1.update(nodetoedge_path)
        except cplex.exceptions.CplexError as e:
            print("CPLEX Error", e)
            newFileWriter.writerow(['CPLEX Error'])
            newFileWriter.writerow([e])

        print('Start of SPC Formulation')
        try:
            model_sp_cont = Model(graph, x_v, w_v, num, tol)
            obj, l_1, c,center_node_2,districts_edges_2,districts_nodes_2 = model_sp_cont.solve_poly_cont()
            print('old_SP_Obj')
            print(obj)
            print('Start of Heuristic')
            new_obj=heuristic(graph, center_node_2, districts_edges_2, tol, num, obj)
            print('new_obj')
            print(new_obj)
            lower_bound_w_cuts = c.solution.MIP.get_best_objective()
            relative_gap_w_cuts = c.solution.MIP.get_mip_relative_gap()
            soln_status_w_cuts = c.solution.get_status_string()
            newFileWriter.writerow(['Computation Time_Empty_SP_Cont', 'Objective Function'])
            newFileWriter.writerow([l_1, round(obj, 2)])
            newFileWriter.writerow(['Lower_Bound','Relative Gap','Solution Status'])
            newFileWriter.writerow([round(lower_bound_w_cuts, 2),relative_gap_w_cuts,soln_status_w_cuts])
            nodetoedge_net_dist_all_2 = {}
            nodetoedge_net_path_all_2 = {}
            for o in center_node_2:
                center_node_index = o
                nodetoedge_net_dist, nodetoedge_path = Dijkstras_Algo(graph, center_node_index,
                                                                      districts_edges_2[center_node_index],
                                                                      districts_nodes_2[center_node_index])
                nodetoedge_net_dist_all_2.update(nodetoedge_net_dist)
                nodetoedge_net_path_all_2.update(nodetoedge_path)

            for o in districts_edges_1:
                for j in districts_edges_1[o]:
                    if j not in districts_edges_2[o]:
                        print('Cut Set District')
                        print('edge in Cut Set district not in SP district')
                        print(j)
                        print('nodes_corr')
                        print(graph.nodes_corr[j])
                        print('district number')
                        print('center')
                        print(o)
                        print('distance within Cut Set district')
                        print(nodetoedge_net_dist_all_1[o][j])
                        print('network distance')
                        print(graph.nodetoedge_net_dist.iloc[[o - 1], [j - 1]])
                        print('SP path from_center_' + str(o) + '_toedge_' + str(j)+ '_within district')
                        a=[graph.nodes_corr[l] for l in nodetoedge_net_path_all_1[o][j]]
                        print(a)
                        print('SP path from_center_' + str(o) + '_toedge_' + str(j))
                        b = [graph.nodes_corr[l] for l in eval(graph.nodetoedge_path_dict[o][j])]
                        print(b)
                        print(eval(graph.nodetoedge_path_dict[o][j]))
                for k in districts_edges_2[o]:
                    if k not in  districts_edges_1[o]:
                        print('SP District')
                        print('edge in SP district not in Cut district')
                        print(k)
                        print('nodes_corr')
                        print(graph.nodes_corr[k])
                        print('center')
                        print(o)
                        print('distance within SP district')
                        print(nodetoedge_net_dist_all_2[o][k])
                        print('network distance')
                        print(graph.nodetoedge_net_dist.iloc[[o-1],[k-1]])
                        print('SP path from_center_'+ str(o) + '_toedge_'+str(k))
                        a=[graph.nodes_corr[l] for l in nodetoedge_net_path_all_2[o][k]]
                        print(a)

            total_length_1={}
            for o in center_node_1:  # Calculating the balance for each district
                lenght_list = [graph.euc_dist_edge_index[q] for q in districts_edges_1[o]]
                total_length_1[o] = sum(lenght_list)
            print('total_length_cut_set')
            print(total_length_1)
            total_length_2 = {}
            for o in center_node_2:  # Calculating the balance for each district
                lenght_list = [graph.euc_dist_edge_index[q] for q in districts_edges_2[o]]
                total_length_2[o] = sum(lenght_list)
            print('total_length_sp_set')
            print(total_length_2)
            # print('lenght of edge'+str(graph.nodes_corr[36]))
            # print(graph.euc_dist_edge_index[36])
            # print('lenght of edge' + str(graph.nodes_corr[38]))
            # print(graph.euc_dist_edge_index[38])
            lenght_list = [graph.euc_dist_edge_index[q] for q in graph.edges_list]
            total_length_b=sum(lenght_list)
            UB=(total_length_b/num)*(1+tol)
            LB=(total_length_b/num)*(1-tol)
            print('UB')
            print(UB)
            print('LB')
            print(LB)
        except cplex.exceptions.CplexError as e:
            print("CPLEX Error", e)
            newFileWriter.writerow(['CPLEX Error'])
            newFileWriter.writerow([e])



# For testing purposes
task=['EBD_SP_Cut_Empty','CARP_F15_g_graph.dat',4,0.01]
execute_task(task)