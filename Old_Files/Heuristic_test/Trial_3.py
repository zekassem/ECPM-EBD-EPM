import networkx as nx
import matplotlib.pyplot as plt

coordinates = [(1, 2), (2, 3), (3, 4), (1, 4), (4, 2), (2, 1)]
nodes = range(1, len(coordinates) + 1)  # Generating node labels (1, 2, 3, ...)
G = nx.Graph()
for node, (x, y) in zip(nodes, coordinates):
    G.add_node(node, pos=(x, y))
edges = [(1, 2), (2, 3), (3, 4), (1, 4), (4, 2), (2, 1), (5,6),(6,1),(3,5),(3,6)]
G.add_edges_from(edges)
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', width=2)
plt.title("Road Network Visualization")
plt.show()

