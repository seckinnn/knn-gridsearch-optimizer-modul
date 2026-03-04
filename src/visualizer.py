import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_line_chart(cv_results, param_grid, dataset_name, output_dir="outputs/plots"):
    plt.figure(figsize=(12, 6))
    for metric in param_grid['metric']:
        for weight in ['uniform', 'distance']:
            mean_test_scores = [
                cv_results['mean_test_score'][i]
                for i in range(len(cv_results['params']))
                if cv_results['params'][i]['metric'] == metric and cv_results['params'][i]['weights'] == weight
            ]
            plt.plot(param_grid['n_neighbors'], mean_test_scores, marker='o', label=f'{metric}-{weight}')
    
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title(f'{dataset_name} - KNN Hyperparameter Tuning (Line Chart)')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name.lower().replace(' ','_')}_line_chart.png"))
    plt.show()

def plot_heatmap(cv_results, param_grid, dataset_name, output_dir="outputs/plots"):
    metrics = param_grid['metric']
    weights = ['uniform', 'distance']
    heatmap_data = np.zeros((len(metrics)*len(weights), len(param_grid['n_neighbors'])))
    row_labels = []

    for i, metric in enumerate(metrics):
        for j, weight in enumerate(weights):
            row_idx = i*len(weights) + j
            row_labels.append(f"{metric}-{weight}")
            for k_idx, k in enumerate(param_grid['n_neighbors']):
                for idx, params in enumerate(cv_results['params']):
                    if params['metric'] == metric and params['weights'] == weight and params['n_neighbors'] == k:
                        heatmap_data[row_idx, k_idx] = cv_results['mean_test_score'][idx]

    plt.figure(figsize=(20, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", xticklabels=param_grid['n_neighbors'], yticklabels=row_labels, cmap="viridis")
    plt.title(f"{dataset_name} - KNN Hyperparameter Tuning (Heatmap)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Metric-Weight")
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name.lower().replace(' ','_')}_heatmap.png"))
    plt.show()