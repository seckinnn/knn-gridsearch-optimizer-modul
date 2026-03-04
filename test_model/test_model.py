# tests/test_model.py
import pytest
from main import train_knn

def test_knn_basic():
    X_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]
    X_test = [[1.5], [0.5]]
    
    grid_search, _ = train_knn(X_train, y_train)  
    model = grid_search.best_estimator_           
    predictions = model.predict(X_test)
    
    assert list(predictions) == [0, 0]

def test_knn_different_k():
    X_train = [[0], [1], [2], [3]]
    y_train = [0, 0, 1, 1]
    X_test = [[2.5], [0.5]]
    
    
    grid_search, _ = train_knn(X_train, y_train)  
    model = grid_search.best_estimator_
    predictions = model.predict(X_test)
    
    assert list(predictions) == [1, 0]