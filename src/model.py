import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X, y, cv=10):
    
    param_grid = {
        'n_neighbors': np.arange(1, 31),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', verbose=0)
    grid_search.fit(X, y)
    
    return grid_search, param_grid