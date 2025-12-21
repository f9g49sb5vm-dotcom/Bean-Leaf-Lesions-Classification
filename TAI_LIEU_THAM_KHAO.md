# 📚 TÀI LIỆU THAM KHẢO - BEAN LEAF LESIONS CLASSIFICATION

## 🎯 MỤC ĐÍCH
Tổng hợp TẤT CẢ các nguồn tài liệu từ GitHub, Kaggle, papers, và web để hoàn thiện bài tập lớn môn Khai Phá Dữ Liệu.

---

## 🏆 TOP KAGGLE NOTEBOOKS (Điểm cao nhất)

### 1. Leaf Disease Detection w/Hybrid Model (ViT, SVM) ⭐⭐⭐⭐⭐
**Tác giả:** arturo-bandini-jr (Daniel Band)  
**Medals:** 62 Bronze  
**Accuracy:** 98%+  
**Link:** https://www.kaggle.com/code/banddaniel/leaf-disease-detection-w-hybrid-model-vit-svm

**Điểm nổi bật:**
- Sử dụng Vision Transformer (ViT) để extract features
- Kết hợp với SVM classifier
- Hybrid approach: Deep Learning + Traditional ML
- Rất phù hợp cho bài tập Data Mining vì kết hợp nhiều kỹ thuật

**Kỹ thuật học được:**
```python
- Vision Transformer architecture
- Feature extraction từ pre-trained models
- SVM classification với extracted features
- Hybrid model design
- Performance comparison
```

**Nên áp dụng:**
- Extract features từ CNN/ViT
- Train SVM trên features này
- So sánh với end-to-end deep learning
- Analyze decision boundaries

---

### 2. Bean Leaf Lesions Pytorch ACC 99% ⭐⭐⭐⭐⭐
**Tác giả:** Chihjung Wang  
**Medals:** 27 Bronze  
**Accuracy:** 99%  
**Link:** https://www.kaggle.com/code/chihjungwang/bean-leaf-lesions-pytorch-acc-99

**Điểm nổi bật:**
- PyTorch implementation hoàn chỉnh
- Đạt 99% accuracy
- Code rất clean và dễ hiểu
- Data augmentation strategies hiệu quả

**Kỹ thuật học được:**
```python
- PyTorch best practices
- Effective data augmentation
- Learning rate scheduling
- Model checkpointing
- Training optimization
```

**Nên áp dụng:**
- PyTorch pipeline structure
- Augmentation strategies
- Training loop optimization
- Validation strategy

---

### 3. Bean Leaf Lesions Classification ⭐⭐⭐⭐
**Tác giả:** Akhil Chhibber  
**Votes:** 15  
**Link:** https://www.kaggle.com/code/akhilchibber/bean-leaf-lesions-classification

**Điểm nổi bật:**
- Complete pipeline từ data loading đến evaluation
- Detailed EDA
- Multiple models comparison
- Clear documentation

**Kỹ thuật học được:**
```python
- Complete ML pipeline
- EDA best practices
- Model comparison framework
- Evaluation metrics
```

---

## 🔗 GITHUB REPOSITORIES QUAN TRỌNG

### 1. akhilchibber/Bean-Leaf-Lesions-Classification
**Link:** https://github.com/akhilchibber/Bean-Leaf-Lesions-Classification  
**License:** MIT  
**Stars:** Growing...

**Nội dung:**
- Complete Jupyter notebook
- Data loading and preprocessing
- Model training and evaluation
- Well-documented code

**File quan trọng:**
- `Bean_Leaf_Lesions_Classification.ipynb` - Main notebook
- `README.md` - Documentation
- Code examples cho transfer learning

---

### 2. PyTorch Vision - torchvision
**Link:** https://github.com/pytorch/vision  
**Official:** PyTorch team

**Models có sẵn:**
```python
# ResNet family
torchvision.models.resnet18()
torchvision.models.resnet34()
torchvision.models.resnet50()
torchvision.models.resnet101()
torchvision.models.resnet152()

# EfficientNet family
torchvision.models.efficientnet_b0()
torchvision.models.efficientnet_b1()
...
torchvision.models.efficientnet_b7()

# VGG family
torchvision.models.vgg16()
torchvision.models.vgg19()

# DenseNet family
torchvision.models.densenet121()
torchvision.models.densenet169()
torchvision.models.densenet201()

# Vision Transformer
torchvision.models.vit_b_16()
torchvision.models.vit_b_32()
torchvision.models.vit_l_16()
```

**Cách sử dụng:**
```python
import torch
import torchvision.models as models

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Modify for 3 classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 3)

# Freeze early layers (optional)
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
```

---

### 3. Plant Disease Recognition Repositories

#### spMohanty/PlantVillage-Dataset
**Link:** https://github.com/spMohanty/PlantVillage-Dataset  
**Dataset:** 54,306 images of healthy and diseased plants

**Học được:**
- Large-scale plant disease dataset structure
- Data organization best practices
- Multi-class classification strategies

#### nshivaprasad/plant-disease-recognition
**Link:** https://github.com/nshivaprasad/plant-disease-recognition

**Techniques:**
- CNN architectures for plant disease
- Transfer learning strategies
- Data augmentation for plants

---

## 📊 DATA MINING ALGORITHMS - GitHub Resources

### 1. Clustering Algorithms

#### scikit-learn/scikit-learn
**Link:** https://github.com/scikit-learn/scikit-learn

**Clustering implementations:**
```python
from sklearn.cluster import (
    KMeans,           # K-Means clustering
    DBSCAN,           # Density-based clustering
    AgglomerativeClustering,  # Hierarchical
    MeanShift,        # Mean-shift
    SpectralClustering,  # Spectral clustering
    Birch,            # BIRCH algorithm
    OPTICS            # OPTICS algorithm
)

from sklearn.mixture import GaussianMixture  # GMM
```

**Examples:**
```python
# K-Means clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(features)

# Evaluation metrics
silhouette = silhouette_score(features, labels)
davies_bouldin = davies_bouldin_score(features, labels)

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

# Visualize clusters
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
plt.title('K-Means Clustering Results')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()
```

---

### 2. Association Rules Mining

#### rasbt/mlxtend
**Link:** https://github.com/rasbt/mlxtend  
**Package:** mlxtend

**Association Rules implementations:**
```python
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# FP-Growth (faster)
frequent_itemsets = fpgrowth(df, min_support=0.3, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Add lift and conviction
rules['lift'] = rules['support'] / (rules['antecedent support'] * rules['consequent support'])

# Sort by confidence and lift
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
```

**Ví dụ cho bài tập:**
```python
# Chuyển image features thành binary format
# Ví dụ: dominant colors, texture patterns

# Color dominance (Red, Green, Blue, Yellow, etc.)
color_features = pd.DataFrame({
    'Red_Dominant': [1, 0, 1, ...],
    'Green_Dominant': [1, 1, 0, ...],
    'Yellow_Spots': [0, 1, 1, ...],
    'Brown_Spots': [1, 0, 1, ...],
})

# Texture patterns
texture_features = pd.DataFrame({
    'Smooth': [1, 0, 0, ...],
    'Rough': [0, 1, 1, ...],
    'Spotted': [1, 1, 0, ...],
})

# Combine features
feature_df = pd.concat([color_features, texture_features], axis=1)

# Apply Apriori
frequent_itemsets = apriori(feature_df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Top 10 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
```

---

### 3. Ensemble Methods

#### dmlc/xgboost
**Link:** https://github.com/dmlc/xgboost  
**Package:** XGBoost

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train
xgb_clf.fit(X_train, y_train)

# Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_clf, max_num_features=20)
plt.title('Feature Importance - XGBoost')
plt.tight_layout()
plt.show()

# Predictions
y_pred = xgb_clf.predict(X_test)
```

#### microsoft/LightGBM
**Link:** https://github.com/microsoft/LightGBM

```python
import lightgbm as lgb

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=-1,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

lgb_clf.fit(X_train, y_train)
y_pred = lgb_clf.predict(X_test)
```

---

### 4. Feature Extraction

#### scikit-image/scikit-image
**Link:** https://github.com/scikit-image/scikit-image

**Color Features:**
```python
from skimage import color, feature
import numpy as np

def extract_color_features(image):
    """Extract color histogram features"""
    # RGB histogram
    hist_r = np.histogram(image[:,:,0], bins=256)[0]
    hist_g = np.histogram(image[:,:,1], bins=256)[0]
    hist_b = np.histogram(image[:,:,2], bins=256)[0]
    
    # HSV conversion
    hsv = color.rgb2hsv(image)
    hist_h = np.histogram(hsv[:,:,0], bins=256)[0]
    hist_s = np.histogram(hsv[:,:,1], bins=256)[0]
    hist_v = np.histogram(hsv[:,:,2], bins=256)[0]
    
    # Color moments
    mean_r, std_r = image[:,:,0].mean(), image[:,:,0].std()
    mean_g, std_g = image[:,:,1].mean(), image[:,:,1].std()
    mean_b, std_b = image[:,:,2].mean(), image[:,:,2].std()
    
    return np.concatenate([
        hist_r, hist_g, hist_b,
        hist_h, hist_s, hist_v,
        [mean_r, std_r, mean_g, std_g, mean_b, std_b]
    ])
```

**Texture Features:**
```python
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

def extract_texture_features(image):
    """Extract texture features"""
    # Convert to grayscale
    gray = color.rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)
    
    # GLCM features
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(gray, distances, angles, levels=256, symmetric=True, normed=True)
    
    contrast = greycoprops(glcm, 'contrast').flatten()
    dissimilarity = greycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    energy = greycoprops(glcm, 'energy').flatten()
    correlation = greycoprops(glcm, 'correlation').flatten()
    
    # LBP features
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    
    return np.concatenate([
        contrast, dissimilarity, homogeneity, energy, correlation,
        hist_lbp
    ])
```

---

## 📝 RESEARCH PAPERS VÀ BLOG POSTS

### 1. Deep Learning for Plant Disease Detection
**Papers:**
- "Using Deep Learning for Image-Based Plant Disease Detection" (2016)
- "Plant Disease Detection Using Deep Learning" (2020)
- "Transfer Learning for Plant Disease Classification" (2021)

### 2. Clustering and Data Mining
**Papers:**
- "K-Means Clustering: Algorithm, Applications, Evaluation Methods" (2018)
- "A Survey of Clustering Data Mining Techniques" (2006)
- "Association Rule Mining: A Survey" (2013)

### 3. Ensemble Methods
**Papers:**
- "Ensemble Methods in Machine Learning" (2000) - Thomas G. Dietterich
- "XGBoost: A Scalable Tree Boosting System" (2016)
- "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017)

---

## 🌐 ONLINE COURSES VÀ TUTORIALS

### 1. Coursera
- **Machine Learning by Andrew Ng** - Stanford
- **Deep Learning Specialization** - deeplearning.ai

### 2. Fast.ai
- **Practical Deep Learning for Coders**
- Link: https://course.fast.ai/

### 3. Kaggle Learn
- **Intro to Machine Learning**
- **Computer Vision**
- **Data Visualization**
- Link: https://www.kaggle.com/learn

---

## 💡 CODE TEMPLATES CHO BÀI TẬP

### Template 1: Complete Pipeline
```python
# 1. Data Loading và EDA
from data_loader import load_and_explore_data
train_df, val_df = load_and_explore_data('path/to/data')

# 2. Feature Extraction
from feature_extraction import extract_all_features
features_train = extract_all_features(train_df)
features_val = extract_all_features(val_df)

# 3. Clustering Analysis
from clustering import perform_clustering_analysis
clustering_results = perform_clustering_analysis(features_train)

# 4. Association Rules
from association import mine_association_rules
rules = mine_association_rules(features_train)

# 5. Classification
from classification import train_multiple_classifiers
models = train_multiple_classifiers(features_train, labels_train)

# 6. Ensemble
from ensemble import create_ensemble
ensemble_model = create_ensemble(models)

# 7. Evaluation
from evaluation import comprehensive_evaluation
results = comprehensive_evaluation(ensemble_model, features_val, labels_val)

# 8. Visualization
from visualization import create_all_plots
create_all_plots(results, clustering_results, rules)
```

### Template 2: Experiment Tracking
```python
import mlflow
import numpy as np

def run_experiment(model_name, model, X_train, y_train, X_val, y_val):
    """Track experiments with MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val, average='weighted')
        val_precision = precision_score(y_val, y_pred_val, average='weighted')
        val_recall = recall_score(y_val, y_pred_val, average='weighted')
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return {
            'model_name': model_name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_f1': val_f1
        }

# Run experiments
results = []
for name, model in models.items():
    result = run_experiment(name, model, X_train, y_train, X_val, y_val)
    results.append(result)
```

---

## 🔍 SIMILAR DATASETS TRÊN KAGGLE

### 1. Rice Leaf Disease Dataset
**Link:** https://www.kaggle.com/datasets/  
**Size:** 3,355 images, 8 GB  
**Classes:** Multiple rice diseases

**Sử dụng để:**
- Compare với bean leaf
- Test generalization
- Transfer learning

### 2. Maize Leaf Disease Dataset
**Size:** 8,852 images, 127 MB  
**Classes:** Various maize diseases

### 3. Corn/Maize Leaf Disease
**Size:** 4,188 images, 169 MB  
**Classes:** 4 classes

**Có thể:**
- Train multi-crop model
- Compare disease patterns
- Cross-dataset validation

---

## 📊 EVALUATION METRICS - COMPREHENSIVE

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)

def comprehensive_evaluation(y_true, y_pred, y_pred_proba=None):
    """Comprehensive model evaluation"""
    results = {}
    
    # Basic metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision'] = precision_score(y_true, y_pred, average='weighted')
    results['recall'] = recall_score(y_true, y_pred, average='weighted')
    results['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Advanced metrics
    results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    results['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # Multi-class AUC
    if y_pred_proba is not None:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        results['auc_ovr'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                           average='weighted', multi_class='ovr')
    
    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    results['classification_report'] = classification_report(y_true, y_pred)
    
    return results

# Visualize results
def plot_comprehensive_results(results):
    """Plot all evaluation results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    
    # 2. Metrics Bar Chart
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa']
    values = [results[m] for m in metrics]
    axes[0,1].bar(metrics, values)
    axes[0,1].set_title('Performance Metrics')
    axes[0,1].set_ylim([0, 1])
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. ROC Curve (if available)
    # ... add ROC curve plotting
    
    # 4. Learning Curve
    # ... add learning curve
    
    plt.tight_layout()
    plt.show()
```

---

## 🎯 TỔNG KẾT TÀI LIỆU

### Top Resources phải xem:
1. ✅ **Kaggle Notebooks** (3 notebooks top)
2. ✅ **PyTorch Vision** (torchvision models)
3. ✅ **scikit-learn** (clustering, classification)
4. ✅ **mlxtend** (association rules)
5. ✅ **XGBoost/LightGBM** (ensemble)
6. ✅ **scikit-image** (feature extraction)

### Packages cần cài:
```bash
pip install torch torchvision
pip install scikit-learn scikit-image
pip install xgboost lightgbm
pip install mlxtend
pip install pandas numpy matplotlib seaborn
pip install opencv-python pillow
pip install mlflow  # experiment tracking
pip install shap  # model explanation
pip install yellowbrick  # visualization
```

### Thứ tự thực hiện:
1. **Tuần 1:** Data Mining techniques (clustering, association rules, feature extraction)
2. **Tuần 2:** Classification, ensemble, evaluation, reporting

---

**Chúc bạn hoàn thành tốt bài tập lớn! 📚✨**
