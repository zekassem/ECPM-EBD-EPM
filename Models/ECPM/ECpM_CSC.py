import os
import time
import numpy as np
import collections
import csv
import pandas as pd
import cplex
from itertools import combinations
import sys
import traceback
import random
import networkx as nx
class Graph():
    def __init__(self, prob):
        t_graph_start = time.time()  # TIME
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
        
        self.edge_nodes = {}  # edge_idx (1-based) -> (u, v)
        for key, idx in self.edge_number.items():
            self.edge_nodes[idx] = key  # key is already (min, max)
            
            
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

        self.demand_edge_index = {}  # creating a dictionary that has the length of each edge using the edge index
        for index, i in enumerate(self.from_node_i):
            self.demand_edge_index[index + 1] = self.demand[index]
        
        self.nodetoedge_net_dist = pd.read_csv("nodetoedge_distance_" + str(prob), header=0, index_col=0)
        self.nodetoedge_path = pd.read_csv("nodetoedge_path_" + str(prob), index_col=0, header=0)
        
        self.nodetoedge_path.columns = self.edges_list
        self.nodetoedge_path = self.nodetoedge_path.transpose()
        self.nodetoedge_path_dict = self.nodetoedge_path.to_dict()
        
        
        t_graph_end = time.time()  # TIME
        self.graph_time = round(t_graph_end - t_graph_start, 2)
        

        print(f"Time to create Graph object: {self.graph_time} sec")


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
    
def compute_planar_faces(Graph):

    G = nx.Graph()
    for key, edge_idx in Graph.edge_number.items():
        G.add_edge(key[0], key[1])

    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("Graph is not planar - face inequalities cannot be applied.")

    faces = []
    visited_darts = set()
    spare_faces=[]
    for u in embedding:
        for v in embedding.neighbors_cw_order(u):
            if (u, v) in visited_darts:
                continue

            face_nodes = embedding.traverse_face(u, v)

            # Always mark darts visited regardless of face size
            for a, b in zip(face_nodes, face_nodes[1:] + [face_nodes[0]]):
                visited_darts.add((a, b))

            if len(face_nodes) > 10 or len(face_nodes) < 3:
                continue

            face_edges = set()
            for a, b in zip(face_nodes, face_nodes[1:] + [face_nodes[0]]):
                key = (min(a, b), max(a, b))
                if key in Graph.edge_number:
                    face_edges.add(Graph.edge_number[key])
            
            if len(face_edges) >= 3 and len(face_edges) <= 10:
                faces.append(frozenset(face_edges))
    return faces  
    
def constraints_wo_cuts(problem, Graph, no_dist,tol, x_v, w_v, use_balance_constraints, cut1, cut2, cut3): #  This function adds all of the constraints for the original districting problem
    num_faces = 0
    num_pvis = 0    
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
    
    
    
    expr = [cplex.SparsePair([x_v[idx], w_v[Graph.node_i[idx] - 1]], [1, -1])
        for idx in range(len(x_v))]
    sens = ["L"] * len(expr)
    rhs = [0] * len(expr)
    problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    # Cut 1:
    # For every node i, if i is selected as a center (w_i = 1), then at least two
    # incident edges must be assigned to territory i:
    #   sum_{e in delta(i)} x_{i,e} >= 2 w_i
    if cut1:
        expr = []
        for i in Graph.nodes_list:
            incident_edges = Graph.edges[i]
            x_incident = [x_v[(i - 1) * Graph.no_edges + (e - 1)] for e in incident_edges]
            expr.append(cplex.SparsePair([w_v[i - 1]] + x_incident,
                                         [-2] + [1] * len(incident_edges)))
        sens = ["G"] * len(expr)
        rhs = [0] * len(expr)
        problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)

    
    # Cut 2:
    # sum_{e in delta(i)} x_{i,e} >= |delta(i)| w_i - sum_{j in N(i)} |delta(i)| w_j
    if cut2:
        expr = []
        for i in Graph.nodes_list:
            incident_edges = Graph.edges[i]
            neighbors = sorted(set(Graph.adj[i]))
            deg_i = len(incident_edges)
    
            x_incident = [x_v[(i - 1) * Graph.no_edges + (e - 1)] for e in incident_edges]
            w_neighbors = [w_v[j - 1] for j in neighbors]
    
            vars_i = x_incident + [w_v[i - 1]] + w_neighbors
            coeff_i = [1] * len(x_incident) + [-deg_i] + [deg_i] * len(w_neighbors)
    
            expr.append(cplex.SparsePair(vars_i, coeff_i))
    
        sens = ["G"] * len(expr)
        rhs = [0] * len(expr)
        problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
    
    if cut3:
        PVI_start = time.time()
        problem, num_faces, num_pvis = add_face_pair_VI(problem, Graph, x_v, w_v)
        PVI_end = time.time()
        print(f"The preprocessing time for generating PVIs is {round(PVI_end - PVI_start, 2)}")
    # Balancing Constraints
    # sum e e E le xie <= sum(le)/p * (1+tau) wi for i e V
    if use_balance_constraints:
        rhs_1 = (sum(Graph.demand_edge_index.values()) / no_dist) * (1 + tol)
    
        expr = [cplex.SparsePair([w_v[i - 1]] + x_v[(i - 1) * Graph.no_edges:i * Graph.no_edges],
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

    return problem, num_faces, num_pvis



def path_structure(edge_set, edge_nodes):
    """Is `edge_set` a single simple path? Returns (is_path, endpoints, interior)."""
    deg = collections.Counter()
    adj = collections.defaultdict(list)
    for e in edge_set:
        a, b = edge_nodes[e]
        deg[a] += 1
        deg[b] += 1
        adj[a].append(e)
        adj[b].append(e)

    seed = next(iter(edge_set))
    seen, stack = {seed}, [seed]
    while stack:
        e = stack.pop()
        for n in edge_nodes[e]:
            for f in adj[n]:
                if f not in seen:
                    seen.add(f)
                    stack.append(f)
    connected = len(seen) == len(edge_set)

    ends = {n for n, d in deg.items() if d == 1}
    interior = {n for n, d in deg.items() if d == 2}
    is_path = connected and max(deg.values()) <= 2 and len(ends) == 2
    return is_path, ends, interior


def add_face_pair_VI(problem, Graph, x_v, w_v):
    r"""
    Adds the corridor inequalities

        x_{i,e} >= sum_{e' in B} x_{i,e'} - |B| + 1 - sum_{j in V(P), j != i} w_j
        for all e in P, for all i,

    where P is a corridor (a u-v path whose interior vertices have degree two
    in G) and B = (delta(u) U delta(v)) \ P.

    Corridor selection. For each pair of bounded faces sharing a path, the
    three u-v connections of the theta configuration are formed (the shared
    path and the two remaining boundary portions). A connection is ELIGIBLE
    if it is a simple u-v path whose interior vertices have degree two in G.
    Among the eligible ones, the two longest are retained, and each is kept
    only if it is NON-DOMINATED: strictly longer than the shortest of the
    other two connections, so that a center can reach both endpoints without
    traversing the corridor and the inequality is not already implied by the
    shortest-path contiguity constraints.

    |B| cap: 3 for all instances, 4 for F15 (the smallest network, whose
    non-dominated cut set is empty at 3). Face size window: 3..10 edges (see
    compute_planar_faces).
    """
    faces = compute_planar_faces(Graph)
    expr, sens, rhs = [], [], []

    b_cap = 4 if "F15" in str(getattr(Graph, "file", "")) else 3

    node_edges = {i: set() for i in Graph.nodes_list}
    for e in range(1, Graph.no_edges + 1):
        u, v = Graph.edge_nodes[e]
        node_edges[u].add(e)
        node_edges[v].add(e)

    edge_len = {e: float(Graph.edge_lengths[e - 1])
                for e in range(1, Graph.no_edges + 1)}

    seen_corridors = set()
    n_paths = 0

    for F1, F2 in combinations(faces, 2):
        shared_edges = F1 & F2
        sym_diff = F1 ^ F2
        if not shared_edges or len(sym_diff) < 2:
            continue

        N_perimeter = set()
        for e_idx in sym_diff:
            u, v = Graph.edge_nodes[e_idx]
            N_perimeter.add(u)
            N_perimeter.add(v)
        N_P = set()
        for e_idx in shared_edges:
            u, v = Graph.edge_nodes[e_idx]
            N_P.add(u)
            N_P.add(v)

        endpoints = N_P & N_perimeter
        if len(endpoints) != 2:
            continue
        u_end, v_end = sorted(endpoints)

        # the three u-v connections of the theta configuration
        trio = [shared_edges, F1 - shared_edges, F2 - shared_edges]
        lens = [sum(edge_len[e] for e in C) for C in trio]

        # eligible corridors: simple u-v paths with interior degree two in G
        elig = []
        for C, LC in zip(trio, lens):
            if not C:
                continue
            is_path, ends, interior = path_structure(C, Graph.edge_nodes)
            if not is_path or ends != endpoints:
                continue
            if any(len(node_edges[n]) != 2 for n in interior):
                continue
            elig.append((LC, C))

        # keep the two longest eligible, each only if non-dominated
        elig.sort(key=lambda t: -t[0])
        for LC, C in elig[:2]:
            other_lens = [lens[j] for j in range(3) if trio[j] is not C]
            if not other_lens or LC <= min(other_lens) + 1e-9:
                continue                      # dominated: a shorter route exists

            key = frozenset(C)
            if key in seen_corridors:
                continue
            seen_corridors.add(key)

            B = list((node_edges[u_end] | node_edges[v_end]) - C)
            B_size = len(B)
            if B_size == 0 or B_size > b_cap:
                continue

            N_C = set()
            for e_idx in C:
                a2, b2 = Graph.edge_nodes[e_idx]
                N_C.add(a2)
                N_C.add(b2)

            n_paths += 1
            for i in Graph.nodes_list:
                base = (i - 1) * Graph.no_edges
                B_indices = [base + (e - 1) for e in B]
                N_C_excl_i = [j for j in N_C if j != i]

                tail_vars = ([x_v[j] for j in B_indices]
                             + [w_v[j - 1] for j in N_C_excl_i])
                tail_coeff = [-1] * B_size + [1] * len(N_C_excl_i)

                for e in C:
                    expr.append(cplex.SparsePair(
                        [x_v[base + (e - 1)]] + tail_vars,
                        [1] + tail_coeff))
                    sens.append("G")
                    rhs.append(1 - B_size)

    if expr:
        problem.linear_constraints.add(lin_expr=expr, senses=sens, rhs=rhs)
        print(f"  [Corridor VI] |B|<={b_cap}: added {len(expr)} constraints "
              f"from {n_paths} non-dominated corridors ({len(faces)} faces).")
    else:
        print(f"  [Corridor VI] |B|<={b_cap}: no non-dominated corridors "
              f"among {len(faces)} faces.")
    return problem, n_paths, len(expr)



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
    def __init__(self,Graph, x_v, w_v,num,tol, use_balance_constraints, cut1, cut2, cut3):
        t_model_start = time.time()  # TIME
        self.x_v=x_v
        self.w_v=w_v
        self.Graph=Graph
        c_org = cplex.Cplex()
        self.no_threads = 'unlimited'
        # c_org.parameters.threads.set(self.no_threads)
        c_org.parameters.mip.strategy.file.set(3)
        # Set a local directory for nodefiles
        # local_nodefile_path = r'C:\Users\zizo_\Desktop\Python\Python\EBD\Models\EBD\Branch_Cut\nodefiles'
        local_nodefile_path = r'/share/uscis/bjabbar/EBD/nodefiles'
        # Ensure the directory exists
        os.makedirs(local_nodefile_path, exist_ok=True)
        c_org.parameters.workdir.set(local_nodefile_path)
        # Setting the objective function to be Minmization
        c_org.objective.set_sense(c_org.objective.sense.minimize)
        # Declare decision variables (first argument is decision variables names, second argument is type of decision variables,
        # third argument is objective function coefficients)
        nodetoedge_net_dist_list = Graph.nodetoedge_net_dist.stack().tolist()  # converting the dataframe into a list
        c_org.variables.add(names=x_v, types=["B"] * len(x_v), obj=nodetoedge_net_dist_list)
        c_org.variables.add(names=w_v, types=["B"] * len(w_v))
        c_main, self.num_faces, self.num_pvis =constraints_wo_cuts(c_org, Graph, num,tol, x_v, w_v, use_balance_constraints, cut1, cut2, cut3)
        self.c_main=c_main
        t_model_end = time.time()  # TIME
        self.model_time = round(t_model_end - t_model_start, 2)
        print(f"Time to set up Model: {self.model_time} sec")
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
        print(f"Time to Solve Using Branch and Cut: {l_1} sec")
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
    t_task_start = time.time() #TIME
    print(f"Model Name {task[0]}")
    print(f"Instance Name {task[1]}")
    print(f"No. of Districts {task[2]}")
    print(f"Balance Tolerance {task[3]}")
    print(f"Use Balance Constraints {task[4]}")
    print(f"Valid Inequaltiy One {task[5]}")
    print(f"Valid Inequaltiy Two {task[6]}")
    print(f"Valid Inequaltiy Three {task[7]}")
    prob=task[1]
    num=task[2]
    tol=task[3]
    use_balance_constraints = task[4]
    cut1=task[5]
    cut2=task[6]
    cut3=task[7]
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
    model_bbb = Model(graph, x_v, w_v, num, tol, use_balance_constraints, cut1, cut2, cut3)
    no_threads = model_bbb.no_threads
    #model_sp_cont = Model(graph, x_v, w_v, num, tol,use_balance_constraints, cut1, cut2, cut3)
    #no_threads = model_sp_cont.no_threads
    with open('cscr/Cut_Empty' + str(num) +'_tol_'+str(tol)+ '_csc4n_' + str(prob) + '_balance_' + str(use_balance_constraints) + '_cut1_' + str(cut1) + '_cut2_' + str(cut2)  + '_PVI_' + str(cut3) + '.csv',
              'w') as newFile:
        print('Directory')
        print('cscr/Cut_Empty' + str(num) +'_tol_'+str(tol)+ '_csc4n_' + str(prob) + '_balance_' + str(use_balance_constraints) + '_cut1_' + str(cut1) + '_cut2_' + str(cut2) + '_PVI_' + str(cut3)  + '.csv')
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
        t_task_end = time.time()  # TIME
        task_time = round(t_task_end - t_task_start, 2)
        print(f"Total Time for Task Execution: {task_time} sec")
        #print('Start of SPC Formulation')
        #try:
        #    obj, l_1, c = model_sp_cont.solve_poly_cont()
        #    lower_bound_w_cuts = c.solution.MIP.get_best_objective()
        #    relative_gap_w_cuts = c.solution.MIP.get_mip_relative_gap()
        #    soln_status_w_cuts = c.solution.get_status_string()
        #    newFileWriter.writerow(['Computation Time_Empty_SP_Cont', 'Objective Function'])
        #    newFileWriter.writerow([l_1, round(obj, 2)])
        #    newFileWriter.writerow(['Lower_Bound','Relative Gap','Solution Status','Num of Faces','Num of PVIs'])
        #    newFileWriter.writerow([round(lower_bound_w_cuts, 2),relative_gap_w_cuts,soln_status_w_cuts,model_sp_cont.num_faces, model_sp_cont.num_pvis])
        #except cplex.exceptions.CplexError as e:
        #    print("CPLEX Error", e)
        #    newFileWriter.writerow(['CPLEX Error'])
        #    newFileWriter.writerow([e])


if __name__ == "__main__":
    # List of tasks: [Model Name, Problem File Name, Number of Districts, Balance Tolerance, Use Balance Constraints, cut1, cut2, cut3]
    t_main_start = time.time() #TIME
    task_list = [
        ["EBD_SP_Cut_Empty", "CARP_O12_g_graph.dat", 2, 0.01, False, True, False, True],
        ["EBD_SP_Cut_Empty", "CARP_O12_g_graph.dat", 10, 0.01, False, True, False, True],
        ["EBD_SP_Cut_Empty", "CARP_O12_g_graph.dat", 30, 0.01, False, True, False, True],
        ["EBD_SP_Cut_Empty", "CARP_O12_g_graph.dat", 40, 0.01, False, True, False, True],
        ["EBD_SP_Cut_Empty", "CARP_O12_g_graph.dat", 50, 0.01, False, True, False, True],
        ["EBD_SP_Cut_Empty", "CARP_O12_g_graph.dat", 100, 0.01, False, True, False, True]
        # Add more tasks here as needed
    ]

    for task in task_list:
        try:
            execute_task(task)
        except Exception as e:
            print(f"Error while executing task {task}: {e}")
    
    t_main_end = time.time()  # TIME
    main_time = round(t_main_end - t_main_start, 2)
    print(f"Total Script Execution Time: {main_time} sec")