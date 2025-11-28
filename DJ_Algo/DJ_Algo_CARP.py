import collections
import copy
import pandas as pd
import numpy as np
import time
import csv
import os
# Loading Data

class Graph():
    def __init__(self,prob):
        self.file = prob
        self.no_nodes = np.loadtxt(self.file, skiprows=1, max_rows=1, usecols=1,
                                   dtype=int)  # skipping lines to get the number of nodes in the planar graph
        self.no_nodes1 = int(self.no_nodes)
        self.no_edges = np.loadtxt(self.file, skiprows=2, max_rows=1, usecols=1,
                                   dtype=int)  # skipping lines to get the
        self.no_edges1 = int(self.no_edges)
        self.adj_list=np.loadtxt(self.file,skiprows=9,max_rows=self.no_edges1,usecols=(2,3),dtype=int)
        self.nodes_list = list(range(1, (self.no_nodes) + 1))
        self.from_node_i = self.adj_list[:, 0].tolist()
        self.to_node_j = self.adj_list[:, 1].tolist()
        self.edges_list = list(range(1, self.no_edges + 1))  # Edges Vector
        # # Finding incident edges for every node
        self.edges = collections.defaultdict(list)
        for i in range(len(self.from_node_i)):
            self.edges[self.from_node_i[i] + 1].append(i + 1)
            self.edges[self.to_node_j[i] + 1].append(i + 1)

        self.edge_lengths= np.loadtxt(self.file,skiprows=9,max_rows=self.no_edges1,usecols=4,dtype=int)
        # This block calculates the length of each edge
        index=0
        self.euc_dist = collections.defaultdict(
            dict)  # dictionary to store the Euclidean Distance of each edge, since it is a symmetric matrix we add the below two lines
        for ind, j in enumerate(self.from_node_i):
            self.euc_dist[self.from_node_i[ind] + 1][self.to_node_j[index] + 1] =self.edge_lengths[index]
            self.euc_dist[self.to_node_j[ind] + 1][self.from_node_i[index] + 1] =self.edge_lengths[index]
            index+=1

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
            self.nodes_corr[i+1]=sorted(self.nodes_corr[i+1])

        t0=time.time()
        self.edge_number=collections.defaultdict(list)
        for key,value in self.nodes_corr.items():
            self.edge_number[(value[0],value[1])]=key

        self.d_1=round(time.time()-t0,2)
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

def Dijkstras_Algo(Graph): # Lazy Implementation of Dijkstra's Algorithm
    t0 = time.time()
    network_dist=collections.defaultdict(dict) # nested dictionary to store the network distance from and to each node
    pred=collections.defaultdict(dict) # nested dictionary to store the predecessor of each node on the shortest path
    for index,source in enumerate(Graph.nodes_list):
        unvisited_nodes=copy.copy(Graph.nodes_list) #list of unvisited_nodes
        for i,j in enumerate(Graph.nodes_list):
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
    l_1=round(time.time() - t0, 2)
    print('dijkstra time')
    print(l_1)
    t0 = time.time()
    # This for loop for printing the shortest path between every two nodes
    path_nodes = collections.defaultdict(dict)  # Nested Dictionary that contains the shortest path in terms of nodes
    for node_index, node in enumerate(Graph.nodes_list):
        for target_index, target_node in enumerate(Graph.nodes_list):
            path_list = []
            u = target_node
            path_list.insert(0, u)
            while pred[node][u] != 0:  # Backtracking the Shortest Path
                path_list.insert(0, pred[node][u])
                u = pred[node][u]
            path_nodes[node][target_node] = path_list

    l_2 = round(time.time() - t0, 2)
    print('Creating Path between nodes in terms of nodes')
    print(l_2)
    t0 = time.time()
    path_edges={i:{j:[graph.edge_number.get((sorted([path_nodes[i][j][k],path_nodes[i][j][k+1]])[0],sorted([path_nodes[i][j][k],path_nodes[i][j][k+1]])[1])) for k,l in enumerate(path_nodes[i][j]) if k + 1 < len(path_nodes[i][j])] for j in Graph.nodes_list} for i in Graph.nodes_list}
    l_3 = round(time.time() - t0, 2)
    print('Creating Path between nodes in terms of edges')
    print(l_3)
    # getting node to edge distance after getting the shortest path between every two nodes
    nodetoedge_net_dist = collections.defaultdict(dict) # Create a nested dictionary to store the distance between each node and edge
    nodetoedge_path=collections.defaultdict(dict) # Create a nested dictionary to store the path between each node and edge
    time_list=[]
    t0 = time.time()
    # creating the min node-to-edge distances nested dictionary, node-to-edge path nested dictionary:
    for i, j in enumerate(Graph.node_i):
        nodes_corr= Graph.nodes_corr[Graph.edge_e[i]]  # getting the corresponding nodes for each edge
        network_dist_1=network_dist[Graph.node_i[i]][nodes_corr[0]] #network distance between the node_i and the first incident node in edge_e
        network_dist_2=network_dist[Graph.node_i[i]][nodes_corr[1]]#network distance between the node_i and the second incident node in edge_e
        nodetoedge_net_dist[Graph.node_i[i]][Graph.edge_e[i]]=min(network_dist_1,network_dist_2) #calculating the minimum of the two distances
        t0=time.time()
        if network_dist_1<network_dist_2:
            nodetoedge_path[Graph.node_i[i]][Graph.edge_e[i]]=path_edges[Graph.node_i[i]][nodes_corr[0]]
        else:
            nodetoedge_path[Graph.node_i[i]][Graph.edge_e[i]]=path_edges[Graph.node_i[i]][nodes_corr[1]]
        t1=round(time.time() - t0, 2)
        time_list.append(t1)
    l_4 = round(time.time() - t0, 2)
    print('Calculating the nodetoedge distance and nodetoedge path')
    print(l_4)
    extra_time=Graph.d_1+l_2+l_3+l_4
    print('total_extra_time')
    print(extra_time)

    return nodetoedge_net_dist,nodetoedge_path,extra_time

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

probs_list=['CARP_F15_g_graph.dat', 'CARP_N17_g_graph.dat', 'CARP_K13_g_graph.dat', 
            'CARP_N16_g_graph.dat', 'CARP_S11_g_graph.dat', 'CARP_K9_p_graph.dat',
            'CARP_N11_g_graph.dat', 'CARP_N15_g_graph.dat', 'CARP_K9_g_graph.dat']

for prob in probs_list:
    # Use absolute path for the input file
    prob_path = os.path.join(script_dir, prob)
    graph=Graph(prob_path)
    nodetoedge_net_dist,nodetoedge_path,extra_time=Dijkstras_Algo(graph)
    df=pd.DataFrame(nodetoedge_net_dist).transpose() # Gets the transpose of the dataframe
    df.to_csv(os.path.join(script_dir, "nodetoedge_distance_"+str(prob)+".csv"),header=True,index=True) # Spits out the shortest path distance from each node (row index) to each edge (column index)
    df_2=pd.DataFrame(nodetoedge_path).transpose() # Gets the transpose of the dataframe
    df_2.to_csv(os.path.join(script_dir, "nodetoedge_path_"+str(prob)+".csv")) # Spits out the shortest path from each node (row index) to each edge (column index)
    # Ensure Results directory exists
    results_dir = os.path.join(script_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'time_print_path_'+str(prob)+'.csv'),
              'w') as newFile:
        newFileWriter = csv.writer(newFile, lineterminator='\n')
        newFileWriter.writerow(['Extra_time'])
        newFileWriter.writerow([extra_time])









