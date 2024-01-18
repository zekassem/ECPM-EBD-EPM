import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
G = nx.Graph()

# Add edges with edge numbers and lengths
edges = [(1, 2, 5), (2, 3, 7), (3, 4, 3)]

# Add edges to the graph
G.add_weighted_edges_from(edges)
# Compute node positions using a layout algorithm
pos = nx.spring_layout(G)
import matplotlib.pyplot as plt

# Draw nodes
nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue')

# Draw edges with labels (edge lengths)
edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Show the graph
plt.show()
