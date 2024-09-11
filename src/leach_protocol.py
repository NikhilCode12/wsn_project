import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def leach_protocol(nodes_positions, cluster_centers, node_energy, num_rounds=10):
    num_nodes = nodes_positions.shape[0]
    num_clusters = cluster_centers.shape[0]

    for _ in range(num_rounds):
        # Assign nodes to the nearest cluster center
        cluster_assignments = np.argmin(np.linalg.norm(nodes_positions[:, np.newaxis] - cluster_centers, axis=2), axis=1)

        # Update energy levels
        for cluster_id in range(num_clusters):
            cluster_nodes = np.where(cluster_assignments == cluster_id)[0]
            if len(cluster_nodes) > 0:
                energy_cost = 1  # Adjusted energy consumption per round
                node_energy[cluster_nodes] -= energy_cost / len(cluster_nodes)
                node_energy[node_energy < 0] = 0  # Ensure energy doesn't go below 0

    return node_energy, cluster_assignments

def plot_leach_results(nodes_positions, cluster_centers, node_energy, cluster_assignments, filename):
    df = pd.DataFrame(nodes_positions, columns=['X', 'Y'])
    df['Energy'] = node_energy
    df['Cluster'] = cluster_assignments

    cluster_df = pd.DataFrame(cluster_centers, columns=['X', 'Y'])
    cluster_df['Cluster'] = range(len(cluster_centers))

    plt.figure(figsize=(14, 8))

    # Plot nodes with color based on energy levels
    scatter = sns.scatterplot(x='X', y='Y', hue='Energy', palette='coolwarm', size='Cluster', sizes=(100, 200), data=df, legend='full', edgecolor='w', marker='o')
    
    # Plot cluster centers with different colors and larger markers
    sns.scatterplot(x='X', y='Y', data=cluster_df, color='red', marker='X', s=250, label='Cluster Centers')

    # Draw lines from each node to its nearest cluster center
    for i, node in enumerate(nodes_positions):
        plt.plot([node[0], cluster_centers[cluster_assignments[i]][0]], [node[1], cluster_centers[cluster_assignments[i]][1]], 'k--', alpha=0.6)

    plt.title('LEACH Clustering Results with Energy Levels', fontsize=18, fontweight='bold')
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    result_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(result_dir, exist_ok=True)  # Create directory if it does not exist
    file_path = os.path.join(result_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')  # Save the plot as a file
    plt.close()  # Close the figure to free memory

def calculate_energy_efficiency(node_energy, num_nodes):
    return np.mean(node_energy == 0)  # Ratio of nodes with 0 energy

def plot_accuracy_curve(contamination_levels, efficiencies, filename):
    plt.figure(figsize=(12, 7))
    plt.plot(contamination_levels, efficiencies, linestyle='-', marker='o', color='teal', linewidth=2, markersize=8)
    plt.ylim(0, 1)
    plt.title('Energy Efficiency vs. Contamination Level in LEACH with Segmentation and IDS', fontsize=18, fontweight='bold')
    plt.xlabel('Contamination Level', fontsize=14)
    plt.ylabel('Energy Efficiency (Proportion of Nodes with 0 Energy)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.annotate(f'{efficiencies[-1]:.2f}', 
                 xy=(contamination_levels[-1], efficiencies[-1]), 
                 xytext=(contamination_levels[-1], efficiencies[-1] + 0.05),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12, color='black')
    
    result_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

def detect_anomalies_isolation_forest(data, contamination):
    isolation_forest = IsolationForest(contamination=contamination)
    predictions = isolation_forest.fit_predict(data)
    return predictions == -1

def detect_anomalies_one_class_svm(data, contamination):
    svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="auto")
    predictions = svm.fit_predict(data)
    return predictions == -1

# Example usage
if __name__ == "__main__":
    from aco_clustering import generate_random_positions_and_energy, run_custom_aco

    NUM_NODES = 100
    DIMENSION = 2
    NUM_CLUSTERS = 5
    ACO_PARAMS = {
        'alpha': 1,
        'beta': 2,
        'rho': 0.5,
        'q': 1,
        'iterations': 100
    }

    contamination_levels = [0.01, 0.02, 0.03, 0.04, 0.05]  # Example contamination levels
    efficiencies = []

    for contamination in contamination_levels:
        nodes_positions, node_energy = generate_random_positions_and_energy(NUM_NODES, DIMENSION)
        best_cluster_centers = run_custom_aco(nodes_positions, node_energy, NUM_CLUSTERS, ACO_PARAMS)
        updated_energy, cluster_assignments = leach_protocol(nodes_positions, best_cluster_centers, node_energy)
        efficiency = calculate_energy_efficiency(updated_energy, NUM_NODES)
        efficiencies.append(efficiency)
    
    plot_leach_results(nodes_positions, best_cluster_centers, updated_energy, cluster_assignments, 'leach_results.png')
    plot_accuracy_curve(contamination_levels, efficiencies, 'leach_accuracy_curve.png')

    # Apply anomaly detection algorithms
    scaler = StandardScaler()
    scaled_positions = scaler.fit_transform(nodes_positions)
    print("Isolation Forest Anomalies:", detect_anomalies_isolation_forest(scaled_positions, 0.05))
    print("One-Class SVM Anomalies:", detect_anomalies_one_class_svm(scaled_positions, 0.05))
