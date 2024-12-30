import collections
import copy
import pandas as pd
import numpy as np
# Loading Data

class Graph():
    def __init__(self,prob):
        self.file = 'data_'+str(prob) + '.csv'
        self.no_nodes = np.loadtxt(self.file, skiprows=1, max_rows=1, usecols=0,
                                   dtype=int)  # skipping lines to get the number of nodes in the planar graph
        self.no_nodes1 = int(self.no_nodes)
        self.coordinates= np.loadtxt(self.file,skiprows=4,max_rows=self.no_nodes1,usecols=(1,2), delimiter=',') # skipping lines to get the x & y coordinates of the nodes
        self.x_coordinate = self.coordinates[:, 0].tolist()
        self.y_coordinate = self.coordinates[:, 1].tolist()
        self.s = self.no_nodes + 4 + 2  # Calculating the lines needed to be skipped to get the number of edges
        self.no_edges = np.loadtxt(self.file, skiprows=self.s, max_rows=1, usecols=0,
                                   dtype=int)  # skipping lines to get the
        self.no_edges1 = int(self.no_edges)
        self.z = self.s + 4  # Calculating the lines needed to be skipped to get adjacency list
        self.adj_list=np.loadtxt(self.file,skiprows=self.z,max_rows=self.no_edges1,usecols=(0,1),dtype=int, delimiter=',')
        self.node_list = list(range(1, (self.no_nodes) + 1))
        self.from_node_i = self.adj_list[:, 0].tolist()
        self.to_node_j = self.adj_list[:, 1].tolist()
        self.edges_list = list(range(1, self.no_edges + 1))  # Edges Vector

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

        # creating a vector of index for every pair of node (i) and edge (e)
        self.node_i = []
        self.edge_e = []
        for i in self.node_list:
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
        for i in self.node_list:
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

# Breadth First Function to make sure that the network is connected
def BFS(Graph):
    # defining the arrays that will be in the algorithm
    pred = []  # predecessor vector
    color = []  # color vector, 0 means white, 1 means gray ,2 means black
    d = []  # distance vector
    Q = []  # set of gray vectors
    s = Graph.node_list[0]  # the source node in the node vector
    for i in Graph.node_list:
        color.append(int(0))  # having all the colors of nodes to be white
        d.append(int(0))
        pred.append(int(0))
    color[s - 1] = 1
    Q.append(s)
    current_dis = 0
    while len(Q) != 0:  # while the cardinality of set of gray nodes is not equal to zero
        u = Q.pop(0)  # Dequeue the first node
        node_nei=Graph.adj[u] # Neighboring nodes
        for i in node_nei:  # This is the main loop
            if color[i - 1] == 0:
                color[i - 1] = 1
                d[i - 1] = current_dis + 1
                pred[i - 1] = u
                Q.append(i)  # appending the set of gray nodes

        color[u - 1] = 2

    b = color
    return b



def Dijkstras_Algo(Graph): # Lazy Implementation of Dijkstra's Algorithm
    network_dist=collections.defaultdict(dict) # nested dictionary to store the network distance from and to each node
    pred=collections.defaultdict(dict) # nested dictionary to store the predecessor of each node on the shortest path
    for index,source in enumerate(Graph.node_list):
        unvisited_nodes=copy.copy(Graph.node_list) #list of unvisited_nodes
        for i,j in enumerate(Graph.node_list):
            network_dist[source][j]=float('inf')  # Adding the infinity label to all nodes
            pred[source][j]=0 #Adding 0 predecessor to all nodes
        network_dist[source][source]=0
        while unvisited_nodes:
            dist_unvisited={k:network_dist[source][k] for k in unvisited_nodes} #getting the distances of unvisited nodes
            node_min=min(dist_unvisited,key=dist_unvisited.get) #getting the node of min. distance within the unvisited nodes
            unvisited_nodes.remove(node_min) #remove the unvisited node with minimum distance
            neigh_nodes=Graph.adj[node_min] # list of neighboring nodes
            for l,k in enumerate(neigh_nodes):
                new_dist=network_dist[source][node_min]+Graph.euc_dist[node_min][k]
                if new_dist<network_dist[source][k]:
                    network_dist[source][k]=new_dist
                    pred[source][k]=node_min

    # This for loop for printing the shortest path between every two nodes
    path_nodes = collections.defaultdict(dict)  # Nested Dictionary that contains the shortest path in terms of nodes
    for node_index, node in enumerate(Graph.node_list):
        for target_index, target_node in enumerate(Graph.node_list):
            path_list = []
            u = target_node
            path_list.insert(0, u)
            while pred[node][u] != 0:  # Backtracking the Shortest Path
                path_list.insert(0, pred[node][u])
                u = pred[node][u]
            path_nodes[node][target_node] = path_list

    # This block creates the shortest path between each two nodes based on edges defined [i,j]
    for i in Graph.node_list:
        for j in Graph.node_list:
            if i == j:
                path_nodes[i][j] = []


    return network_dist,path_nodes

probs_list=[51]
for prob in probs_list:
    graph=Graph(prob)
    nodetonode_net_dist,nodetonode_path=Dijkstras_Algo(graph)
    df=pd.DataFrame(nodetonode_net_dist).transpose() # Gets the transpose of the dataframe
    df.to_csv("nodetonode_distance_"+str(prob)+".csv",header=True,index=True) # Spits out the shortest path distance from each node (row index) to each node (column index)
    df_2=pd.DataFrame(nodetonode_path).transpose() # Gets the transpose of the dataframe
    df_2.to_csv("nodetonode_path_"+str(prob)+".csv") # Spits out the shortest path from each node (row index) to each node (column index)
    print(df_2.shape)







