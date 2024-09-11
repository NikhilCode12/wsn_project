import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Parameters
NUM_NODES = 100
NUM_CLUSTERS = 5
DIMENSION = 2

# ACO Parameters
ACO_PARAMS = {
    'alpha': 1,
    'beta': 2,
    'rho': 0.5,
    'q': 1,
    'iterations': 100
}

# Generate random positions for nodes and initial energy levels
def generate_random_positions_and_energy(num_nodes, dimension):
    positions = np.random.rand(num_nodes, dimension)
    energy = np.random.rand(num_nodes)  # Random initial energy levels
    return positions, energy

# Custom ACO class with dynamic segmentation
class CustomAntColonyOptimizer:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, q):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

    def optimize(self, objective_function, n_clusters, nodes_positions, node_energy):
        best_solution = None
        best_fitness = float('inf')

        # Initialize pheromone matrix
        pheromone = np.ones((n_clusters, DIMENSION))

        for iteration in range(self.num_iterations):
            solutions = []
            fitnesses = []

            # Each ant constructs a solution
            for _ in range(self.num_ants):
                cluster_centers = np.random.rand(n_clusters, DIMENSION)
                fitness = objective_function(cluster_centers, nodes_positions, node_energy)
                solutions.append(cluster_centers)
                fitnesses.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = cluster_centers

            # Update pheromones
            pheromone *= (1 - self.rho)
            for i, solution in enumerate(solutions):
                pheromone += self.q / fitnesses[i]

        return best_solution

# Fitness function considering both distance and residual energy
def fitness_function(cluster_centers, nodes_positions, node_energy):
    total_distance = 0
    total_energy_penalty = 0
    for node, energy in zip(nodes_positions, node_energy):
        distances = np.linalg.norm(node - cluster_centers, axis=1)
        closest_distance = np.min(distances)
        total_distance += closest_distance
        # Penalize based on residual energy
        energy_penalty = (1 - energy) * closest_distance
        total_energy_penalty += energy_penalty
    return total_distance + total_energy_penalty

# Run custom ACO to find optimal cluster centers
def run_custom_aco(nodes_positions, node_energy, num_clusters, aco_params):
    aco = CustomAntColonyOptimizer(
        num_ants=num_clusters,
        num_iterations=aco_params['iterations'],
        alpha=aco_params['alpha'],
        beta=aco_params['beta'],
        rho=aco_params['rho'],
        q=aco_params['q']
    )
    
    def objective_function(cluster_centers, nodes_positions, node_energy):
        return fitness_function(cluster_centers, nodes_positions, node_energy)
    
    best_solution = aco.optimize(objective_function, n_clusters=num_clusters, 
                                 nodes_positions=nodes_positions, node_energy=node_energy)
    
    return best_solution

# Plot and save clustering results with seaborn
def plot_results(nodes_positions, cluster_centers, node_energy, filename):
    df = pd.DataFrame(nodes_positions, columns=['X', 'Y'])
    df['Energy'] = node_energy
    df['Cluster'] = -1  # Initialize cluster column with -1 (unassigned)

    # Create a DataFrame for cluster centers
    cluster_df = pd.DataFrame(cluster_centers, columns=['X', 'Y'])
    cluster_df['Cluster'] = range(len(cluster_centers))

    plt.figure(figsize=(14, 8))

    # Plot nodes with color based on energy levels
    scatter = plt.scatter(df['X'], df['Y'], c=df['Energy'], cmap='viridis', s=100, edgecolor='k', label='Nodes')
    plt.colorbar(scatter, label='Residual Energy')  # Add color bar to show residual energy levels

    # Plot cluster centers with different colors
    colors = sns.color_palette('husl', NUM_CLUSTERS)  # Use a distinct color palette
    for i, center in enumerate(cluster_centers):
        plt.scatter(center[0], center[1], color=colors[i], marker='X', s=200, edgecolor='k', label=f'Cluster Head {i+1}')
        plt.text(center[0], center[1], f'CH{i+1}', color='black', fontsize=12, ha='center', va='center')

    # Draw lines from each node to its nearest cluster center
    for i, node in enumerate(nodes_positions):
        distances = np.linalg.norm(node - cluster_centers, axis=1)
        closest_cluster = np.argmin(distances)
        plt.plot([node[0], cluster_centers[closest_cluster][0]], [node[1], cluster_centers[closest_cluster][1]], 'k--', alpha=0.5)

    plt.title('Clustering Based on Dynamic Segmentation')
    plt.legend(loc='upper right')
    
    result_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(result_dir, exist_ok=True)  # Create directory if it does not exist
    file_path = os.path.join(result_dir, filename)
    plt.savefig(file_path)  # Save the plot as a file
    plt.close()  # Close the figure to free memory

    df = pd.DataFrame(nodes_positions, columns=['X', 'Y'])
    df['Energy'] = node_energy
    df['Cluster'] = -1  # Initialize cluster column with -1 (unassigned)

    # Create a DataFrame for cluster centers
    cluster_df = pd.DataFrame(cluster_centers, columns=['X', 'Y'])
    cluster_df['Cluster'] = range(len(cluster_centers))

    plt.figure(figsize=(14, 8))

    # Plot nodes with color based on energy levels
    sns.scatterplot(x='X', y='Y', hue='Energy', palette='viridis', size='Cluster', sizes=(100, 200), data=df, legend=None)

    # Plot cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, edgecolor='k', label='Cluster Centers')

    # Draw lines from each node to its nearest cluster center
    for i, node in enumerate(nodes_positions):
        distances = np.linalg.norm(node - cluster_centers, axis=1)
        closest_cluster = np.argmin(distances)
        plt.plot([node[0], cluster_centers[closest_cluster][0]], [node[1], cluster_centers[closest_cluster][1]], 'k--', alpha=0.5)

    # Annotate cluster centers
    for i, center in enumerate(cluster_centers):
        plt.text(center[0], center[1], f'Cluster {i}', color='red', fontsize=12, ha='right', va='bottom')

    plt.title('Clustering Based on Dynamic Segmentation')
    plt.legend()
    
    result_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(result_dir, exist_ok=True)  # Create directory if it does not exist
    file_path = os.path.join(result_dir, filename)
    plt.savefig(file_path)  # Save the plot as a file
    plt.close()  # Close the figure to free memory

# Plot and save energy distribution chart
def plot_energy_distribution(node_energy, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(node_energy, bins=20, color='teal', edgecolor='black')
    plt.xlabel('Residual Energy')
    plt.ylabel('Number of Nodes')
    plt.title('Distribution of Residual Energy Levels')
    
    result_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(result_dir, exist_ok=True)  # Create directory if it does not exist
    file_path = os.path.join(result_dir, filename)
    plt.savefig(file_path)  # Save the plot as a file
    plt.close()  # Close the figure to free memory

# Main execution
if __name__ == "__main__":
    nodes_positions, node_energy = generate_random_positions_and_energy(NUM_NODES, DIMENSION)
    best_cluster_centers = run_custom_aco(nodes_positions, node_energy, NUM_CLUSTERS, ACO_PARAMS)
    print("Optimal Cluster Centers:")
    print(best_cluster_centers)
    
    # Plot and save the clustering results
    plot_results(nodes_positions, best_cluster_centers, node_energy, 'cluster_plot_100nodes_seaborn2.png')
    
    # Plot and save the energy distribution chart
    plot_energy_distribution(node_energy, 'energy_distribution2.png')
