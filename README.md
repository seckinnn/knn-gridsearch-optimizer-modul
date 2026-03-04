# KNN Classifier with GridSearchCV (Modular Version)

Bu proje, Python ve scikit-learn kullanarak **KNN algoritmasını farklı veri setlerinde (Iris, Digits, Breast Cancer) test eder** ve **GridSearchCV ile en iyi hiperparametreleri bulur**.  

---

## 🚀 Özellikler

✅ Hiperparametre optimizasyonu (`n_neighbors`, `metric`, `weights`)  
✅ Confusion matrix, classification report, accuracy score  
✅ Sonuçların görselleştirilmesi: line chart ve heatmap  
✅ Sonuçların JSON rapor olarak kaydedilmesi  

---

## 🧩 Kullanılan Teknolojiler ve Paketler

| Paket           | Açıklama |
|-----------------|----------|
| numpy            | Matematiksel işlemler için |
| pandas           | Veri setlerini yükleme ve düzenleme |
| matplotlib       | Line chart görselleştirmeleri için |
| seaborn          | Heatmap görselleştirmeleri için |
| scikit-learn     | KNN ve GridSearchCV için |
| os               | Dosya ve klasör yönetimi |

---

## 🏗️ Modüler Klasör Yapısı


knn-algorithm-ml/
│
├─ src/
│   ├─ data_loader.py      # Veri setlerini yükler
│   ├─ evaluator.py        # Model eğitimi ve GridSearchCV
│   ├─ model.py            # KNN modeli oluşturma
│   ├─ visualizer.py       # Line chart ve heatmap fonksiyonları
│   └─ __init__.py
│
├─ outputs/
│   ├─ plots/              # Line chart ve heatmap görselleri
│   └─ reports/            # JSON formatında sonuçlar
│
├─ main.py                 # Projenin çalıştırılabilir ana dosyası
├─ requirements.txt        # Gerekli paketler
└─ README.md