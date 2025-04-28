import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(28)

# Create a modular network: assume 3 modules
modules = []
G = nx.Graph()

# Parameters
num_modules = 4
nodes_per_module = 10
p_intra = 0.7  # High probability of intra-module connection
p_inter = 0.05  # Low probability of inter-module connection

# Create nodes for each module
node_offset = 0
for m in range(num_modules):
    module_nodes = list(range(node_offset, node_offset + nodes_per_module))
    G.add_nodes_from(module_nodes)
    modules.append(module_nodes)

    # Add intra-module edges
    for i in module_nodes:
        for j in module_nodes:
            if i < j and np.random.rand() < p_intra:
                G.add_edge(i, j)
    node_offset += nodes_per_module

# Add inter-module edges
for m1 in range(num_modules):
    for m2 in range(m1 + 1, num_modules):
        for i in modules[m1]:
            for j in modules[m2]:
                if np.random.rand() < p_inter:
                    G.add_edge(i, j)

# Draw the network
pos = nx.spring_layout(G, seed=28)  # force-directed layout, input=Graph, output=dict of each node positions

colors = ['red', 'blue', 'green', 'purple']
node_colors = []
for idx, module in enumerate(modules):
    node_colors += [colors[idx]] * len(module)

plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.title("Modular Connectivity Graph")
plt.axis('off')
plt.show()


# Parameters for dynamics
num_steps = 100  # How many time steps to simulate
threshold = 0.  # Firing threshold
initial_active_prob = 0.1  # Initial probability that a neuron is active

# Initialize node states randomly: 1 = active, 0 = inactive
np.random.seed(None)
node_states = {node: (1 if np.random.rand() < initial_active_prob else 0) for node in G.nodes}

# Record the activity over time
activity_over_time = []

for step in range(num_steps):
    new_states = {}
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            new_states[node] = node_states[node]
            continue

        # Compute average activity of neighbors
        neighbor_activity = np.mean([node_states[neighbor] for neighbor in neighbors])

        # Simple rule: activate if neighbor activity is high enough
        if neighbor_activity > threshold:
            new_states[node] = 1
        else:
            new_states[node] = 0

    node_states = new_states
    # Save current activity
    activity_over_time.append([node_states[node] for node in G.nodes])

# Convert to numpy array for easier plotting
activity_array = np.array(activity_over_time)

# Plot the activity over time
plt.figure(figsize=(10, 6))
plt.imshow(activity_array.T, aspect='auto', cmap='binary', interpolation='nearest')
plt.xlabel('Time step')
plt.ylabel('Neuron (node)')
plt.title('Neural Activity Dynamics Over Time')
plt.colorbar(label='Activity (0 = inactive, 1 = active)')
plt.show()
