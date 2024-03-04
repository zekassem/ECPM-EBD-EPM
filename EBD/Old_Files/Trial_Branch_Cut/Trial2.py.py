import numpy as np
import csv
import cplex
import collections
from cplex.exceptions import CplexError
import time
from operator import itemgetter, attrgetter
import copy
from numpy import median
import math
# has the two new logic cuts
# Importing .dat file
prob=51
data=str(prob)+".dat"
no_nodes=np.loadtxt(data,skiprows=1,max_rows=1,usecols=0,dtype=int) # skipping lines to get the number of nodes in the planar graph
no_nodes1=int(no_nodes)
coordinates= np.loadtxt(data,skiprows=4,max_rows=no_nodes1,usecols=(1,2)) # skipping lines to get the x & y coordinates of the nodes
s=no_nodes+4+2 # Calculating the lines needed to be skipped to get the number of edges
no_edges=np.loadtxt(data,skiprows=s,max_rows=1,usecols=0,dtype=int) # skipping lines to get the
no_edges1=int(no_edges)
z=s+3 # Calculating the lines needed to be skipped to get adjacency list
adj_list=np.loadtxt(data,skiprows=z,max_rows=no_edges1,usecols=(0,1),dtype=int)
q=s+3+no_edges+2 # Calculating the lines needed to be skipped to get the number of districts
x_coordinate=coordinates[:,0].tolist()
y_coordinate=coordinates[:,1].tolist()
node=list(range(1,(no_nodes)+1))
from_node_i=adj_list[:,0].tolist()
to_node_j=adj_list[:,1].tolist()

print(cplex.__version__)
edges_v=list(range(1,no_edges+1)) #Edges Vector
# Input parameters to the districting problem
no_dist=[2] # the number of districts
t=0.5 # tolerance parameter into the balancing constraints



# creating a vector of index for every pair of node (i) and edge (e)
node_i = []
edge_e = []
for i in node:
    for j in edges_v:
        node_i.append(i)
        edge_e.append(j)


# End of Data
# Creating Variables

# Creating a list of variable names (x_i,(j,k)): binary variable whether the edge (j,k) is assigned to district i
x_v = []
for i in range(len(node_i)):
    x = 'x' + str(node_i[i]) + '_' + str(edge_e[i])
    x_v.append(x)




# Creating a list of variable names (wi): binary variable whether the node i is the center or not
w_v = []
for i in range(len(node)):
    w = 'w' + str(node[i])
    w_v.append(w)





# importing the distance from every node to every edge
d_nodetoedge=np.loadtxt(open("nodetoedge_djalg"+str(prob)+".csv", "rb"), delimiter=",", skiprows=1,max_rows=len(node_i),usecols=2).tolist()


dist_v = []
for i in range(len(from_node_i)):
    dist = ((x_coordinate[from_node_i[i]] - x_coordinate[to_node_j[i]]) ** 2 + (y_coordinate[from_node_i[i]] - y_coordinate[to_node_j[i]]) ** 2) ** 0.5
    dist_v.append(dist)


# Finding Neighboring nodes for each node (using only one loop)
adj=collections.defaultdict(list)
for i in range(len(from_node_i)):
    adj[from_node_i[i]+1].append(to_node_j[i]+1)
    adj[to_node_j[i]+1].append(from_node_i[i]+1)

# Finding incident edges for every node
edges=collections.defaultdict(list)
for i in range(len(from_node_i)):
    edges[from_node_i[i]+1].append(i+1)
    edges[to_node_j[i]+1].append(i+1)


# Creating a list of the degree for every node
degree=[]
for i in node:
    degree.append(len(edges[i]))


# Creating a dictionary for neighboring edges for each edge
edges_neighboring= collections.defaultdict(list)
for e in edges_v:
    nodes_corr=[]
    nodes_corr.append(from_node_i[e - 1]+1)
    nodes_corr.append(to_node_j[e - 1]+1)
    edges_nei=[]
    for i in nodes_corr:
        edges_nei.extend(edges[i])
    edges_nei1=list(set(edges_nei))
    l=[]
    l.append(e)
    edges_nei=[x for x in edges_nei1 if x not in l]
    edges_neighboring[e].extend(edges_nei)


# Breadth First Function
def BFS(edges_v,index, result, l):
    # defining the arrays that will be in the algorithm
    pred = []  # predecessor vector
    color = []  # color vector, 0 means white, 1 means gray ,2 means black
    d = []  # distance vector
    Q = []  # set of gray vectors
    s = l  # the source edge index in the edge vector
    for e in edges_v:
        color.append(int(0))  # having all the colors of edges to be white
        d.append(int(0))
        pred.append(int(0))

    color[s - 1] = 1
    Q.append(s)
    current_dis = 0

    while len(Q) != 0:  # while the cardinality of set of gray edges is not equal to zero
        u = Q.pop(0)  # Dequeue the first edge
        edges_nei=edges_neighboring[u] # Neighboring edges

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
def branch_cut(a,center_node,result,edges_v,c): # this function takes the solution from relaxed problem and checks for connected components and then add cuts as needed # input a,center_node, result, edges_v,c (optimization problem)
    while a > 0:
        C = []
        index = 0
        result = [[1, 6], [2, 3, 4, 5, 7]]
        for o in center_node:  # This loop is to detect whether there are disconnected districts or not
            explored_edges_s1 = []  # edges that are found using BFS
            R = []  # Set of Edges that need to be explored using BFS
            R.extend(result[index])
            l = R[0]  # the source edge from which we will start exploring
            b = BFS(edges_v, index, result, l)
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
                print('explored_edges_new_s1')
                print(explored_edges_s1)
                print('explored_edges')
                print(explored_edges_s)
                # Find the connected component
                # Find the neighboring edges to the connected component (excluding edges in the connected component)
                # Add the needed cuts
                connect_edges = [j for i, j in enumerate(R) if b[j - 1] == 2]  # Connected Component (Set Sk)
                connect_edges_neighboring1 = [j for j in edges_neighboring[i] for i in connect_edges]
                connect_edges_neighboring = set(connect_edges_neighboring1)  # Removing duplicated edges
                connect_edges_neighboring_n = list(connect_edges_neighboring.difference(
                    set(
                        connect_edges)))  # Neighboring edges to connected component excluding edges in the connected component
                s = (1 - len(connect_edges))  # 1-|S|
                sum_x = []
                coeff = []
                indicies_1 = [(o - 1) * no_edges + (q - 1) for q in connect_edges]
                x_variables = itemgetter(*indicies_1)(x_v)
                if len(connect_edges) == 1:
                    sum_x.append(x_variables)
                else:
                    sum_x.extend(x_variables)
                coeff1 = [-1 for r in range(len(indicies_1))]
                coeff.extend(coeff1)
                indicies_2 = [(o - 1) * no_edges + (q - 1) for q in connect_edges_neighboring_n]
                x_variables_1 = itemgetter(*indicies_2)(x_v)
                coeff2 = [1 for r in range(len(indicies_2))]
                coeff.extend(coeff2)
                if len(connect_edges_neighboring_n) == 1:
                    sum_x.append(x_variables_1)
                else:
                    sum_x.extend(x_variables_1)
                c.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["G"], rhs=[s])
                if len(unexplored_edges) > 0:  # finding the next connected component
                    l = unexplored_edges[0]
                    b = BFS(edges_v, index, result, l)
            index = index + 1
        if sum(C) < len(center_node):
            a = 1
            c.solve()
            center_node1 = [i for i, val in enumerate(c.solution.get_values(w_v)) if val > 0]
            center_node = [i + 1 for i in center_node1]
            result = []
            for o in center_node:
                result.append([edge_e[i] for i, val in enumerate(c.solution.get_values(x_v)) if val > 0 and node_i[i] == o])
        else:
            a = 0
    return result,c #returning the c after adding all of the cuts (object c)

def constraints_wo_cuts(c,edges_v, no_nodes,no_edges,no_dist,t): #  This function adds all of the constraints for the original districting problem
    # input edges_v, no_nodes,no_edges,no_dist,t
    # sum i e V (xie)=1 for e e E: each edge is assigned to one territory
    for e in edges_v:
        sum_x = x_v[(e - 1):no_edges * no_nodes:no_edges]
        coeff = [1 for i in range(no_nodes)]
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["E"], rhs=[1])
        sum_x = []
        coeff = []

    # sum i e V w_i=p
    sum_x = w_v
    coeff = [1 for i in range(no_nodes)]
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["E"],
                             rhs=[no_dist])
    sum_x = []
    coeff = []
    sum_x2 = []
    # Balancing Constraints
    # sum e e E xie <= (|E|/p) * (1+tau) wi for i e V
    rhs_1 = (sum(dist_v) / no_dist) * (1 + t)
    for i in node:
        sum_x.append(w_v[i - 1])
        coeff.append(-rhs_1)
        sum_x.extend(x_v[(i - 1) * no_edges:i * no_edges])
        coeff2 = [q for q in dist_v]
        coeff.extend(coeff2)
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["L"], rhs=[0])
        coeff = []
        sum_x = []

    # sum e e E xie >= (|E|/p) * (1-tau) wi for i e V
    sum_x = []
    coeff = []
    coeff2 = []
    rhs_2 = (sum(dist_v) / no_dist) * (1 - t)
    for i in node:
        sum_x.append(w_v[i - 1])
        sum_x.extend(x_v[(i - 1) * no_edges:i * no_edges])
        coeff.append(-rhs_2)
        coeff2 = [q for q in dist_v]
        coeff.extend(coeff2)
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(sum_x, coeff)], senses=["G"], rhs=[0])
        coeff = []
        sum_x = []


for num in no_dist:
    distr_v = list(range(1, num + 1))  # Districts vector
    with open('Results/computation_time_1st_Logic_no_dist_'+str(num)+'_prob_'+str(prob)+'.csv','w') as newFile:
        newFileWriter = csv.writer(newFile, lineterminator='\n')
        newFileWriter.writerow(['Computation Time_WO_Logic_Cuts', 'Objective Function'])
        try:
            # Declare CPLEX object
            c = cplex.Cplex()
            # Setting the objective function to be Minmization
            c.objective.set_sense(c.objective.sense.minimize)
            # Declare decision variables (first argument is decision variables names, second argument is type of decision variables,
            # third argument is objective function coefficients)
            c.variables.add(names=x_v, types=["B"] * len(x_v), obj=d_nodetoedge)
            c.variables.add(names=w_v, types=["B"] * len(w_v))
            constraints_wo_cuts(c, edges_v, no_nodes, no_edges, num, t)
            t0 = time.time()
            c.solve()
        except CplexError as exc:
            print (exc)
        l_1 = round(time.time() - t0,2)
        newFileWriter.writerow([l_1])
        t_1=time.time()
        center_node1 = [i for i, val in enumerate(c.solution.get_values(w_v)) if val > 0]
        center_node = [i + 1 for i in center_node1]
        result=[]
        for o in center_node:
            result.append([edge_e[i] for i,val in enumerate(c.solution.get_values(x_v)) if val>0 and node_i[i]==o])
        a = 1
        # input to the function a, center_node,result, edges_v,c, output: obj, c.solution
        result=[[1,3],[2,4,5,6,7]]
        result,c=branch_cut(a,center_node,result,edges_v,c)
        obj = c.solution.get_objective_value()
        l_2 = round(time.time() - t_1,2)
        newFileWriter.writerow([l_2])
        newFileWriter.writerow([l_2+l_1,round(obj,2)])
        sol_1 = [x_v[i] for i,j in enumerate(c.solution.get_values(x_v)) if j>0]
        sol_2 = [w_v[i] for i,j in enumerate(c.solution.get_values(w_v)) if j>0]
        print(sol_1)
        print(sol_2)
        print(obj)

