from sklearn.datasets import load_iris, load_digits, load_breast_cancer

def load_datasets():
    """
    Üç veri setini sözlük halinde döndürür.
    """
    datasets = {
        "Iris": load_iris(),
        "Digits": load_digits(),
        "Breast Cancer": load_breast_cancer()
    }
    return datasets