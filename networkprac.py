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

# Initialize neuron activity with random firing rates between 0 and 1
neuron_activity = np.random.rand(len(G.nodes))

# Parameters for the model
threshold = 0.5  # Firing threshold (for simplicity)
alpha = 0.1  # How much influence neighbors have on the activity
num_steps = 100  # Number of time steps for simulation

# Activity history (for plotting)
activity_history = []

# Simulate neural activity dynamics over time
for t in range(num_steps):
    new_activity = neuron_activity.copy()

    # Update each neuron's activity based on its neighbors
    for i in G.nodes:
        # Get the sum of activity from neighboring neurons
        neighbor_activity = np.mean([neuron_activity[j] for j in G.neighbors(i)])

        # Simple rule: increase activity based on neighbor activity
        new_activity[i] = neuron_activity[i] + alpha * neighbor_activity

        # Apply threshold: if activity is too high, neuron fires (value capped at 1)
        if new_activity[i] > threshold:
            new_activity[i] = 1
        else:
            new_activity[i] = 0

    neuron_activity = new_activity
    activity_history.append(neuron_activity)

# Convert activity history to a numpy array for plotting
activity_array = np.array(activity_history)

# Plot the neural activity dynamics
plt.imshow(activity_array.T, aspect='auto', cmap='binary', interpolation='nearest')
plt.colorbar(label="Activity (0 = inactive, 1 = active)")
plt.title("Neural Activity Dynamics Over Time")
plt.xlabel("Time step")
plt.ylabel("Neuron (node)")
plt.show()


