from sklearn.model_selection import train_test_split
from src.data_loader import load_datasets
from src.model import train_knn
from src.evaluator import evaluate_model
from src.visualizer import plot_line_chart, plot_heatmap

datasets = load_datasets()

for name, dataset in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    
    grid_search, param_grid = train_knn(X_train, y_train)
    
    print(f"{name} - Best Params: {grid_search.best_params_}")
    
    evaluate_model(grid_search.best_estimator_, X_test, y_test, name)
    
    plot_line_chart(grid_search.cv_results_, param_grid, name)
    plot_heatmap(grid_search.cv_results_, param_grid, name)