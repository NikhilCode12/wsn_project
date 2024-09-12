import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
import os

# Import functions from other scripts
from leach_protocol import leach_protocol, plot_leach_results
from aco_clustering import generate_random_positions_and_energy, run_custom_aco
from lmnn_optimized import lmnn_distance_optimization, detect_anomalies

def calculate_energy_efficiency(node_energy):
    return np.mean(node_energy == 0)

def calculate_accuracy(true_anomalies, detected_anomalies, num_samples):
    y_true = np.zeros(num_samples)
    y_true[true_anomalies] = 1
    y_pred = np.zeros(num_samples)
    y_pred[detected_anomalies] = 1
    return accuracy_score(y_true, y_pred) * 100

def plot_accuracy_curve(contaminations, accuracies, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(contaminations, accuracies, marker='o', linestyle='-', color='teal')
    plt.title('Accuracy vs. Contamination Level')
    plt.xlabel('Contamination Level')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(filename, format='png')
    plt.close()

def plot_energy_efficiency_curve(contaminations, efficiencies, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(contaminations, efficiencies, marker='o', linestyle='-', color='purple')
    plt.title('Energy Efficiency vs. Contamination Level')
    plt.xlabel('Contamination Level')
    plt.ylabel('Energy Efficiency')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(filename, format='png')
    plt.close()

def detect_anomalies_isolation_forest(data, contamination):
    isolation_forest = IsolationForest(contamination=contamination)
    predictions = isolation_forest.fit_predict(data)
    return predictions == -1

def detect_anomalies_one_class_svm(data, contamination):
    svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="auto")
    predictions = svm.fit_predict(data)
    return predictions == -1

def detect_anomalies_local_outlier_factor(data, contamination):
    lof = LocalOutlierFactor(n_neighbors=30, contamination=contamination)
    predictions = lof.fit_predict(data)
    return predictions == -1

if __name__ == "__main__":
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

    contamination_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
    efficiencies = []
    accuracies = []

    for contamination in contamination_levels:
        nodes_positions, node_energy = generate_random_positions_and_energy(NUM_NODES, DIMENSION)
        best_cluster_centers = run_custom_aco(nodes_positions, node_energy, NUM_CLUSTERS, ACO_PARAMS)
        updated_energy, cluster_assignments = leach_protocol(nodes_positions, best_cluster_centers, node_energy)
        efficiency = calculate_energy_efficiency(updated_energy)
        efficiencies.append(efficiency)

        labels = np.random.randint(0, 3, NUM_NODES)
        lmnn_model = lmnn_distance_optimization(nodes_positions, labels)
        true_anomalies = np.random.choice(NUM_NODES, int(NUM_NODES * 0.05), replace=False)
        detected_anomalies = detect_anomalies(nodes_positions, lmnn_model, contamination)
        accuracy = calculate_accuracy(true_anomalies, detected_anomalies, NUM_NODES)
        accuracies.append(accuracy)

    # Save results
    plot_leach_results(nodes_positions, best_cluster_centers, updated_energy, cluster_assignments, 'leach_results_final.png')
    plot_accuracy_curve(contamination_levels, accuracies, 'lmnn_accuracy_curve_final.png')
    plot_energy_efficiency_curve(contamination_levels, efficiencies, 'energy_efficiency_curve_final.png')

    # Apply anomaly detection algorithms
    scaler = StandardScaler()
    scaled_positions = scaler.fit_transform(nodes_positions)
    for contamination in contamination_levels:
        isolation_forest_anomalies = detect_anomalies_isolation_forest(scaled_positions, contamination)
        one_class_svm_anomalies = detect_anomalies_one_class_svm(scaled_positions, contamination)
        lof_anomalies = detect_anomalies_local_outlier_factor(scaled_positions, contamination)
        print(f"Contamination Level {contamination}:")
        print("Isolation Forest Anomalies:", np.sum(isolation_forest_anomalies))
        print("One-Class SVM Anomalies:", np.sum(one_class_svm_anomalies))
        print("Local Outlier Factor Anomalies:", np.sum(lof_anomalies))

    print("Simulation completed and results saved.")
