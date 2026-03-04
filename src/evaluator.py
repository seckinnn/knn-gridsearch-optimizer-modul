from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import os

def evaluate_model(model, X_test, y_test, dataset_name, output_dir="outputs/reports"):
    
    
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    error = 1 - acc
    
    report = {
        "dataset": dataset_name,
        "accuracy": acc,
        "error_rate": error,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"\n{dataset_name} Dataset Evaluation:")
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("Error rate: {:.2f}%".format(error * 100))
    print("Confusion Matrix:\n", report["confusion_matrix"])
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{dataset_name.lower().replace(' ','_')}_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    
    return report