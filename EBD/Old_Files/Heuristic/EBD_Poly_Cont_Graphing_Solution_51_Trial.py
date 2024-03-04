import cplex
import time
import numpy as np
import collections
import csv
import pandas as pd
import itertools
import copy
import networkx as nx
import matplotlib.pyplot as plt

# Loading Graph
class Graph():
    def __init__(self,prob):
        self.file = str(prob) + '.dat'
        self.no_nodes = np.loadtxt(self.file, skiprows=1, max_rows=1, usecols=0,
                                   dtype=int)  # skipping lines to get the number of nodes in the planar graph
        self.no_nodes1 = int(self.no_nodes)
        self.coordinates = np.loadtxt(self.file, skiprows=4, max_rows=self.no_nodes1,
                                      usecols=(1, 2))  # skipping lines to get the x & y coordinates of the nodes
        self.x_coordinate = self.coordinates[:, 0].tolist()
        self.y_coordinate = self.coordinates[:, 1].tolist()
        self.s = self.no_nodes + 4 + 2  # Calculating the lines needed to be skipped to get the number of edges
        self.no_edges = np.loadtxt(self.file, skiprows=self.s, max_rows=1, usecols=0,
                                   dtype=int)  # skipping lines to get the
        self.no_edges1 = int(self.no_edges)
        self.z = self.s + 3  # Calculating the lines needed to be skipped to get adjacency list
        self.adj_list = np.loadtxt(self.file, skiprows=self.z, max_rows=self.no_edges1, usecols=(0, 1), dtype=int)
        self.q = self.s + 3 + self.no_edges + 2  # Calculating the lines needed to be skipped to get the number of districts
        self.nodes_list = list(range(1, (self.no_nodes) + 1))
        self.from_node_i = self.adj_list[:, 0].tolist()
        self.to_node_j = self.adj_list[:, 1].tolist()
        self.coordinates_tuple = list(zip(self.x_coordinate, self.y_coordinate))  # creating a tuple for the x and y-coordinate
        self.from_node_i_plus=[i + 1 for i in self.from_node_i]  # adding one to each node
        self.to_node_j_plus=[i + 1 for i in self.to_node_j]  # adding one to each node
        self.edge_tuple = tuple(zip(self.from_node_i_plus, self.to_node_j_plus))
        self.edges_list = list(range(1, self.no_edges + 1))  # Edges Vector
        self.nodetoedge_net_dist = pd.read_csv("nodetoedge_distance_" + str(prob) + ".csv", header=0, index_col=0)
        self.nodetoedge_path = pd.read_csv("nodetoedge_path_" + str(prob) + ".csv", index_col=0, header=0)
        self.nodetoedge_path.columns = self.edges_list
        self.nodetoedge_path = self.nodetoedge_path.transpose()
        self.nodetoedge_path_dict = self.nodetoedge_path.to_dict()
        # Finding incident edges for every node
        self.edges = collections.defaultdict(list)
        for i in range(len(self.from_node_i)):
            self.edges[self.from_node_i[i] + 1].append(i + 1)
            self.edges[self.to_node_j[i] + 1].append(i + 1)

        # This block calculates the length of each edge
        self.euc_dist = collections.defaultdict(
            dict)  # dictionary to store the Euclidean Distance of each edge, since it is a symmetric matrix we add the below two lines
        for index, j in enumerate(self.from_node_i):
            self.euc_dist[self.from_node_i[index] + 1][self.to_node_j[index] + 1] = ((self.x_coordinate[
                                                                                          self.from_node_i[index]] -
                                                                                      self.x_coordinate[self.to_node_j[
                                                                                          index]]) ** 2 + (
                                                                                             self.y_coordinate[
                                                                                                 self.from_node_i[
                                                                                                     index]] -
                                                                                             self.y_coordinate[
                                                                                                 self.to_node_j[
                                                                                                     index]]) ** 2) ** 0.5
            self.euc_dist[self.to_node_j[index] + 1][self.from_node_i[index] + 1] = ((self.x_coordinate[
                                                                                          self.from_node_i[index]] -
                                                                                      self.x_coordinate[self.to_node_j[
                                                                                          index]]) ** 2 + (
                                                                                             self.y_coordinate[
                                                                                                 self.from_node_i[
                                                                                                     index]] -
                                                                                             self.y_coordinate[
                                                                                                 self.to_node_j[
                                                                                                     index]]) ** 2) ** 0.5

        self.euc_dist_edge_index = {}  # creating a dictionary that has the length of each edge using the edge index
        for index, i in enumerate(self.from_node_i):
            self.euc_dist_edge_index[index + 1] = self.euc_dist[self.from_node_i[index] + 1][self.to_node_j[index] + 1]
        # creating a vector of index for every pair of node (i) and edge (e)
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

        # Corresponding nodes for every edge
        self.nodes_corr = collections.defaultdict(list)
        for i in range(len(self.from_node_i)):
            self.nodes_corr[i + 1].append(self.from_node_i[i] + 1)
            self.nodes_corr[i + 1].append(self.to_node_j[i] + 1)

        # Creating a list of the degree for every node
        self.degree = []
        for i in self.nodes_list:
            self.degree.append(len(self.edges[i]))

        # Creating a dictionary for neighboring edges for each edge (Looks like it can be further simplified)
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


## End of Data
# Input parameters to the districting problem
no_dist_list=[2] # the number of districts
tol_list=[0.5] #Balance Tolerance


def Graphing_Solution(Graph,no_dist,tol,formulation_type,center_node,distircts_edges):
    #Graphing_Solution(graph, num, tol, formulation_type_2, center_node, districts_edges)
    np.random.seed(200)
    # creating p number of colors for each district tuple for (R,G,B)
    district_color = {o:tuple(np.random.rand(3)) for o in center_node} #creating a dictionary with index equal to center node and value is the color of the district
    # creating tuple of zeros
    edge_color_list = [tuple(np.zeros(3)) for i in Graph.edges_list]
    node_color_list = ['salmon' for i in Graph.nodes_list]

    for o in center_node:
        node_color_list[o-1] = district_color[o]

    pos = {}  # creating a dictionary with node: (x_coordinate,y_coordinate)
    for i in Graph.nodes_list:
        pos[i] = Graph.coordinates_tuple[i - 1]

    G = nx.Graph()
    G.add_nodes_from(Graph.nodes_list)
    G.add_edges_from(Graph.edge_tuple)
    districts_edges_ind = collections.defaultdict(list) # creating a dictionary with districts defined in terms of edges [i,j] like this

    for i in distircts_edges:
        for j in distircts_edges[i]:
            districts_edges_ind[i].append(sorted(Graph.nodes_corr[j]))
    for i, edge in enumerate(G.edges()):
        l = sorted([edge[0], edge[1]])
        for j in district_color:
            if l in districts_edges_ind[j]:
                edge_color_list[i] = district_color[j]
    nx.draw(G, pos=pos, with_labels=True, node_size=300, edge_color=edge_color_list, node_color=node_color_list)
    plt.rcParams["figure.figsize"] = (30, 10)
    plt.savefig('Results/'+str(formulation_type)+'_no_dist_'+str(no_dist)+'_tol_'+str(tol)+'_prob_'+str(prob)+'.png', format='png')
    plt.show()





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
            for l,k in enumerate(neigh_nodes):
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


def constraints_wo_cuts(problem, Graph, no_dist,t, x_v, w_v): #  This function adds all of the constraints for the original districting problem
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
    rhs_1 = (sum(Graph.euc_dist_edge_index.values()) / no_dist) * (1 + t)

    expr = [cplex.SparsePair([w_v[i - 1]]+x_v[(i - 1) * no_edges:i * no_edges],[-rhs_1]+list(Graph.euc_dist_edge_index.values())) for i in Graph.nodes_list]
    sens=["L"]*len(expr)
    rhs=[0]*len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    # sum e e E le xie >= sum(le)/p * (1-tau) wi for i e V
    rhs_2 = (sum(Graph.euc_dist_edge_index.values()) / no_dist) * (1 - t)
    expr = [cplex.SparsePair([w_v[i - 1]] + x_v[(i - 1) * no_edges:i * no_edges],
                             [-rhs_2] + list(Graph.euc_dist_edge_index.values())) for i in Graph.nodes_list]
    rhs = [0] * len(expr)
    sens=["G"]*len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    return problem

def logic_cuts(Graph,problem,w_v,x_v): # function that adds the logic cuts
    # Logic Cut sum (i,k) e delta(i) xi,(i,k) >= 2 * wi for i e V, logic cuts are added as user cuts
    expr = [cplex.SparsePair([x_v [(q - 1) + (i - 1) * Graph.no_edges] for q in Graph.edges[i]]+[w_v[i-1]],[1 for r in range(len(Graph.edges[i]))]+[-2]) for i in Graph.nodes_list]
    sens = ["G"] * len(expr)
    rhs = [0] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    return problem

def SP_contiguity_const(Graph,x_v,problem): # Function that adds the contiguity constraints: x_i,(j,k)<= x_i,(l,m) where (l,m) \in SP_i,(j,k) and (l,m) \in Cutset({(j,k)}) \forall i \in V, \forall (j,k) \in E
    expr = [cplex.SparsePair([x_v[index], x_v[(node_i[index]-1)*Graph.no_edges+(eval(Graph.nodetoedge_path[Graph.node_i[index]][Graph.edge_e[index]])[-1]-1)]], [1, -1]) for index, i in enumerate(Graph.node_i) if len(eval(Graph.nodetoedge_path[Graph.node_i[index]][Graph.edge_e[index]]))>0]
    sens = ["L"] * len(expr)
    rhs = [0] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    return problem

class Model(): # This is the model where we add the variables and the constraints
    def __init__(self,Graph, x_v, w_v,num,t):
        self.x_v=x_v
        self.w_v=w_v
        self.Graph=Graph
        c_org = cplex.Cplex()
        # Setting the objective function to be Minmization
        c_org.objective.set_sense(c_org.objective.sense.minimize)
        # Declare decision variables (first argument is decision variables names, second argument is type of decision variables,
        # third argument is objective function coefficients)
        nodetoedge_net_dist_list = Graph.nodetoedge_net_dist.stack().tolist()  # converting the dataframe into a list
        c_org.variables.add(names=x_v, types=["B"] * len(x_v), obj=nodetoedge_net_dist_list)
        c_org.variables.add(names=w_v, types=["B"] * len(w_v))
        c_main=constraints_wo_cuts(c_org, Graph, num,t, x_v, w_v)
        self.c_main=c_main



    def branch_bound_cut(self): # This is the method that has the branch and bound and cut
        x_v = self.x_v
        w_v = self.w_v
        Graph = self.Graph
        c_1=self.c_main
        d=logic_cuts(Graph, c_1, w_v, x_v)
        d.parameters.clocktype.set(2)
        d.parameters.timelimit.set(43200)
        t0 = time.time()
        d.solve()
        result=[]
        center_node1 = [i for i, val in enumerate(d.solution.get_values(w_v)) if val > 0.01]
        center_node = [i + 1 for i in center_node1]
        for o in center_node:
            result.append([Graph.edge_e[i] for i,val in enumerate(d.solution.get_values(x_v)) if val>0.01 and Graph.node_i[i]==o])
        a = 1
        # num_iterations=0 # counter for the number of iterations
        # num_cuts=0 # counter for the number of
        while a > 0:
            # num_iterations+=1
            C = []
            index = 0
            for o in center_node:  # This loop is to detect whether there are disconnected districts or not
                explored_edges_s1 = []  # edges that are found using BFS
                R = []  # Set of Edges that need to be explored using BFS
                R.extend(result[index])
                l = R[0]  # the source edge from which we will start exploring
                b = BFS(Graph, index, result, l)
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
                    indicies_1 = [(o - 1) * no_edges + (q - 1) for q in connect_edges]
                    x_variables = [x_v[i] for i in indicies_1]
                    sum_x.extend(x_variables)
                    coeff1 = [-1 for r in range(len(indicies_1))]
                    coeff.extend(coeff1)
                    indicies_2 = [(o - 1) * no_edges + (q - 1) for q in connect_edges_neighboring_n]
                    x_variables_1 = [x_v[i] for i in indicies_2]
                    coeff2 = [1 for r in range(len(indicies_2))]
                    coeff.extend(coeff2)
                    sum_x.extend(x_variables_1)
                    # if len(sum_x)>0:
                        # num_cuts+=1
                    d.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["G"], rhs=[s])
                    if len(unexplored_edges) > 0:  # finding the next connected component
                        l = unexplored_edges[0]
                        b = BFS(Graph, index, result, l)
                index = index + 1
            if sum(C) < len(center_node):
                a = 1
                d.solve()
                center_node1 = [i for i, val in enumerate(d.solution.get_values(w_v)) if val > 0.01]
                center_node = [i + 1 for i in center_node1]
                result = []
                for o in center_node:
                    result.append(
                        [Graph.edge_e[i] for i, val in enumerate(d.solution.get_values(x_v)) if val > 0.01 and node_i[i] == o])
            else:
                a = 0
        l_1 = round(time.time() - t0, 2)
        sol_1 = [x_v[i] for i, j in enumerate(d.solution.get_values(x_v)) if j > 0.01]
        sol_2 = [w_v[i] for i, j in enumerate(d.solution.get_values(w_v)) if j > 0.01]
        obj = d.solution.get_objective_value()
        print("gap tolerance = ", d.parameters.mip.tolerances.mipgap.get())
        print(sol_1)
        print(sol_2)
        print(obj)
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
        return obj,l_1,d,center_node,districts_edges,districts_nodes

    def solve_poly_cont(self):
        Graph = self.Graph
        x_v = self.x_v
        w_v = self.w_v
        c_2=self.c_main
        c=SP_contiguity_const(Graph, x_v, c_2)
        c=logic_cuts(Graph, c, w_v, x_v)
        c.parameters.clocktype.set(2)
        c.parameters.timelimit.set(43200)
        t0 = time.time()
        c.solve()
        l_1 = round(time.time() - t0, 2)
        sol_1 = [x_v[i] for i, j in enumerate(c.solution.get_values(x_v)) if j > 0.01]
        sol_2 = [w_v[i] for i, j in enumerate(c.solution.get_values(w_v)) if j > 0.01]
        obj = c.solution.get_objective_value()
        center_node = [i+1 for i, val in enumerate(c.solution.get_values(w_v)) if val > 0.01]
        districts_edges={}
        for o in center_node:
            districts_edges[o]=[Graph.edge_e[i] for i, val in enumerate(c.solution.get_values(x_v)) if
                           val > 0.01 and Graph.node_i[i] == o]

        districts_nodes=collections.defaultdict(list)
        for o in center_node:
            districts_nodes[o].extend([Graph.nodes_corr[i] for i in districts_edges[o]])
        for o in center_node:
            districts_nodes[o]=list(set(list(itertools.chain(*districts_nodes[o]))))
        print('Solution')
        print(sol_1)
        print(sol_2)
        print('Objective Function')
        print(obj)
        return obj,l_1,c,center_node,districts_edges,districts_nodes

probs_list=[51]

for prob in probs_list:
    graph = Graph(prob)
    nodes_list=graph.nodes_list
    edges_list=graph.edges_list
    no_nodes=graph.no_nodes
    no_edges=graph.no_edges
    node_i=graph.node_i
    edge_e=graph.edge_e
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

    for num in no_dist_list:
        for tol in tol_list:
            print('Parameters')
            print(f'no of districts{num}')
            distr_v = list(range(1, num + 1))  # Districts vector
            with open('Results/EBD_Cut_Set_vs_SP_Contiguity_Cut_1_no_dist_'+str(num)+'_tol_'+str(tol)+'_prob_'+str(prob)+'.csv','w') as newFile:
                newFileWriter = csv.writer(newFile, lineterminator='\n')
                formulation_type_1='Cut_Set'
                newFileWriter.writerow(['Computation Time_W_BBB', 'Objective Function'])
                model_bbb=Model(graph,x_v, w_v, num,tol)
                obj, l_1, d,center_node,districts_edges,districts_nodes = model_bbb.branch_bound_cut()
                nodetoedge_net_dist_all={} # dictionary that has all of the distances
                for o in center_node:
                    center_node_index=o
                    nodetoedge_net_dist, nodetoedge_path=Dijkstras_Algo(graph,center_node_index,districts_edges[center_node_index],districts_nodes[center_node_index])
                    nodetoedge_net_dist_all.update(nodetoedge_net_dist)

                Graphing_Solution(graph, num, tol, formulation_type_1, center_node, districts_edges)
                lower_bound= d.solution.MIP.get_best_objective()
                newFileWriter.writerow([l_1 , round(obj, 2)])
                newFileWriter.writerow(['Lower_Bound'])
                newFileWriter.writerow([round(lower_bound,2)])
                formulation_type_2 = 'SP_Contiguity'
                model_sp_cont=Model(graph,x_v, w_v, num,tol)
                obj, l_1, c,center_node,districts_edges,districts_nodes= model_sp_cont.solve_poly_cont()
                nodetoedge_net_dist_all_2={}
                for o in center_node:
                    center_node_index=o
                    nodetoedge_net_dist, nodetoedge_path=Dijkstras_Algo(graph,center_node_index,districts_edges[center_node_index],districts_nodes[center_node_index])
                    nodetoedge_net_dist_all_2.update(nodetoedge_net_dist)

                Graphing_Solution(graph, num, tol, formulation_type_2,center_node, districts_edges)
                lower_bound_w_cuts = c.solution.MIP.get_best_objective()
                newFileWriter.writerow(['Computation Time_W_SP_Cont', 'Objective Function'])
                newFileWriter.writerow([l_1, round(obj, 2)])
                newFileWriter.writerow(['Lower_Bound'])
                newFileWriter.writerow([round(lower_bound_w_cuts, 2)])
                for i, (key,value) in enumerate(nodetoedge_net_dist_all.items()):
                    if nodetoedge_net_dist_all[key]!=nodetoedge_net_dist_all_2[key]:
                        print(key)







