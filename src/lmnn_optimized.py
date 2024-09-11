import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from metric_learn import LMNN
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os

def lmnn_distance_optimization(node_positions, labels):
    lmnn = LMNN(n_neighbors=10)  # Increased number of neighbors
    lmnn.fit(node_positions, labels)
    return lmnn

def detect_anomalies(node_positions, lmnn_model, contamination_level):
    transformed_data = lmnn_model.transform(node_positions)
    
    # Feature Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(transformed_data)
    
    # Initialize anomaly detection methods with hyperparameter tuning
    isolation_forest = IsolationForest(contamination=contamination_level, n_estimators=200, max_samples='auto')
    lof = LocalOutlierFactor(n_neighbors=30, contamination=contamination_level)
    ocsvm = OneClassSVM(nu=contamination_level, kernel='rbf', gamma='scale')
    
    # Fit and predict
    isolation_forest_scores = isolation_forest.fit_predict(scaled_data)
    lof_scores = lof.fit_predict(scaled_data)
    ocsvm_scores = ocsvm.fit_predict(scaled_data)

    # Combine results using voting ensemble
    combined_scores = (isolation_forest_scores + lof_scores + ocsvm_scores) / 3
    anomalies = np.where(combined_scores < 0)[0]
    
    return anomalies

def plot_lmnn_results(node_positions, lmnn_model, anomalies, filename):
    transformed_data = lmnn_model.transform(node_positions)
    df = pd.DataFrame(transformed_data, columns=['X', 'Y'])
    df['Anomaly'] = 0
    df.loc[anomalies, 'Anomaly'] = 1
    
    plt.figure(figsize=(14, 8))
    palette = {0: 'blue', 1: 'red'}
    scatter = sns.scatterplot(x='X', y='Y', hue='Anomaly', palette=palette, data=df, s=100, alpha=0.8)
    scatter.legend(title='Anomaly', loc='upper right', frameon=True, shadow=True)
    plt.title('LMNN Results with Anomaly Detection', fontsize=16)
    plt.xlabel('Transformed Feature 1', fontsize=14)
    plt.ylabel('Transformed Feature 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    result_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_accuracy(true_anomalies, detected_anomalies, num_samples):
    y_true = np.zeros(num_samples)
    y_true[true_anomalies] = 1
    
    y_pred = np.zeros(num_samples)
    y_pred[detected_anomalies] = 1
    
    return accuracy_score(y_true, y_pred) * 100

def plot_accuracy_curve(contaminations, accuracies, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(contaminations, accuracies, linestyle='-', color='teal', linewidth=2)
    plt.ylim(0, 100)
    plt.title('Accuracy vs. Contamination Level', fontsize=16)
    plt.xlabel('Contamination Level', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.annotate(f'{accuracies[-1]:.2f}%', 
                 xy=(contaminations[-1], accuracies[-1]), 
                 xytext=(contaminations[-1], accuracies[-1] + 5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    
    result_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, filename)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    def generate_random_positions_and_energy(num_nodes, dimension):
        np.random.seed(42)
        nodes_positions = np.random.rand(num_nodes, dimension) * 100
        labels = np.random.randint(0, 3, num_nodes)
        num_anomalies = int(num_nodes * 0.05)
        true_anomalies = np.random.choice(num_nodes, num_anomalies, replace=False)
        return nodes_positions, labels, true_anomalies

    NUM_NODES = 100
    DIMENSION = 2

    nodes_positions, labels, true_anomalies = generate_random_positions_and_energy(NUM_NODES, DIMENSION)
    lmnn_model = lmnn_distance_optimization(nodes_positions, labels)
    
    contamination_levels = [0.01, 0.02, 0.03, 0.04, 0.05] 
    accuracies = []

    for contamination in contamination_levels:
        detected_anomalies = detect_anomalies(nodes_positions, lmnn_model, contamination)
        accuracy = calculate_accuracy(true_anomalies, detected_anomalies, NUM_NODES)
        accuracies.append(accuracy)

    plot_accuracy_curve(contamination_levels, accuracies, 'lmnn_accuracy_curve.png')
