# 🎓 BÀI TẬP LỚN: KHAI PHÁ DỮ LIỆU - PHÂN LOẠI BỆNH LÁ ĐẬU

## 📋 THÔNG TIN BÀI TẬP

**Môn học:** Khai Phá Dữ Liệu (Data Mining)  
**Thời gian:** 2 tuần  
**Dataset:** [Bean Leaf Lesions Classification - Kaggle](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)  
**Mục tiêu:** Xây dựng hệ thống khai phá dữ liệu toàn diện để phân loại bệnh trên lá đậu

---

## 🎯 YÊU CẦU BÀI TẬP LỚN MÔN KHAI PHÁ DỮ LIỆU

### PHẦN 1: KHAI PHÁ VÀ TIỀN XỬ LÝ DỮ LIỆU (30%)

#### 1.1. Thu Thập và Hiểu Dữ Liệu
- ✅ Tải và khám phá dataset từ Kaggle
- ✅ Phân tích cấu trúc và đặc tính dữ liệu
- ✅ Xác định các thuộc tính quan trọng

#### 1.2. Làm Sạch Dữ Liệu (Data Cleaning)
- [ ] **Phát hiện và xử lý giá trị bất thường (Outliers)**
  - Phát hiện ảnh bị lỗi, nhiễu
  - Phát hiện ảnh duplicate
  - Loại bỏ hoặc sửa chữa ảnh không hợp lệ
  
- [ ] **Xử lý dữ liệu thiếu (Missing Data)**
  - Kiểm tra ảnh bị hỏng
  - Xử lý metadata thiếu
  
- [ ] **Chuẩn hóa và tiêu chuẩn hóa**
  - Normalize giá trị pixel (0-1)
  - Standardize kích thước ảnh
  - Chuẩn hóa format (RGB)

#### 1.3. Phân Tích Khám Phá Dữ Liệu (EDA) - Chi Tiết
- [ ] **Phân tích thống kê mô tả**
  - Phân phối số lượng mẫu theo lớp
  - Phân tích mất cân bằng dữ liệu
  - Thống kê kích thước, aspect ratio
  - Phân tích độ phân giải ảnh
  
- [ ] **Phân tích phân phối màu sắc**
  - Histogram của các kênh màu (R, G, B)
  - Phân tích độ sáng, độ tương phản
  - Phân tích spectrum màu của từng lớp
  
- [ ] **Phân tích tương quan**
  - Correlation giữa các đặc trưng màu sắc
  - Phân tích sự khác biệt giữa các lớp
  
- [ ] **Trực quan hóa**
  - Box plots cho phân tích outliers
  - Violin plots cho phân phối
  - Heatmaps cho correlation matrix
  - t-SNE/PCA visualization cho dimensionality reduction

#### 1.4. Trích Xuất Đặc Trưng (Feature Engineering)
- [ ] **Đặc trưng màu sắc (Color Features)**
  ```python
  - Color histograms
  - Color moments (mean, std, skewness)
  - HSV color space features
  - LAB color space features
  - Dominant colors extraction
  ```

- [ ] **Đặc trưng texture (Texture Features)**
  ```python
  - GLCM (Gray Level Co-occurrence Matrix)
  - LBP (Local Binary Patterns)
  - Gabor filters
  - HOG (Histogram of Oriented Gradients)
  - Haralick features
  ```

- [ ] **Đặc trưng hình dạng (Shape Features)**
  ```python
  - Contour features
  - Edge detection features
  - Area, perimeter, compactness
  - Hu moments
  ```

- [ ] **Đặc trưng Deep Learning**
  ```python
  - CNN features extraction
  - Transfer learning features (ResNet, VGG)
  - Autoencoder features
  ```

---

### PHẦN 2: THUẬT TOÁN KHAI PHÁ DỮ LIỆU (40%)

#### 2.1. Phân Cụm (Clustering) - QUAN TRỌNG cho môn Data Mining
- [ ] **K-Means Clustering**
  - Phân cụm ảnh dựa trên đặc trưng màu sắc
  - Xác định số cụm tối ưu (Elbow method, Silhouette score)
  - Phân tích và visualize các cụm
  - So sánh cụm với nhãn thực tế
  
- [ ] **Hierarchical Clustering**
  - Dendrogram visualization
  - Agglomerative clustering
  - So sánh với K-Means
  
- [ ] **DBSCAN**
  - Phát hiện outliers
  - Clustering với mật độ khác nhau
  
- [ ] **Gaussian Mixture Models (GMM)**
  - Soft clustering
  - Phân tích xác suất thuộc cụm

**Code mẫu:**
```python
# K-Means Clustering cho phân tích dữ liệu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Trích xuất features từ ảnh
features = extract_color_texture_features(images)

# Xác định số cụm tối ưu
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features, labels))

# Visualize Elbow method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Số cụm k')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Số cụm k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()

# Áp dụng K-Means với k tối ưu
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(features)

# So sánh cụm với nhãn thực tế
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(true_labels, clusters)
nmi = normalized_mutual_info_score(true_labels, clusters)
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
```

#### 2.2. Luật Kết Hợp (Association Rules Mining)
- [ ] **Apriori Algorithm**
  - Tìm patterns xuất hiện cùng nhau
  - Phân tích đặc trưng xuất hiện trong mỗi lớp bệnh
  
- [ ] **FP-Growth**
  - Frequent Pattern Mining
  - Tìm quy luật giữa các đặc trưng
  
**Code mẫu:**
```python
# Association Rules cho phân tích đặc trưng
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Chuyển features thành binary format
# Ví dụ: màu sắc dominant, texture patterns
feature_binary = binarize_features(features)

# Áp dụng Apriori
frequent_itemsets = apriori(feature_binary, min_support=0.3, use_colnames=True)

# Tạo association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

print("Top 10 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Visualize rules
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.scatterplot(x='support', y='confidence', size='lift', data=rules, sizes=(50, 400))
plt.title('Association Rules Visualization')
plt.show()
```

#### 2.3. Phân Loại (Classification) - Đa dạng thuật toán
- [ ] **Decision Tree**
  - Xây dựng cây quyết định
  - Pruning để tránh overfitting
  - Visualize decision tree
  - Feature importance analysis
  
- [ ] **Random Forest**
  - Ensemble of decision trees
  - Feature importance
  - Out-of-bag error estimation
  
- [ ] **SVM (Support Vector Machine)**
  - Linear SVM
  - RBF kernel SVM
  - Multi-class SVM
  
- [ ] **Naive Bayes**
  - Gaussian Naive Bayes
  - Probabilistic classification
  
- [ ] **k-NN (k-Nearest Neighbors)**
  - Distance-based classification
  - Optimal k selection
  
- [ ] **Neural Networks**
  - MLP (Multi-Layer Perceptron)
  - CNN (Convolutional Neural Network)
  - Transfer Learning (ResNet, EfficientNet, VGG)

**Code mẫu:**
```python
# So sánh nhiều thuật toán phân loại
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

# Danh sách các thuật toán
classifiers = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB(),
    'k-NN': KNeighborsClassifier(n_neighbors=5)
}

# Đánh giá từng thuật toán
results = []
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    results.append({
        'Algorithm': name,
        'Mean Accuracy': scores.mean(),
        'Std Accuracy': scores.std(),
        'Min Accuracy': scores.min(),
        'Max Accuracy': scores.max()
    })

# Hiển thị kết quả
results_df = pd.DataFrame(results).sort_values('Mean Accuracy', ascending=False)
print(results_df)

# Visualize comparison
plt.figure(figsize=(12, 6))
plt.bar(results_df['Algorithm'], results_df['Mean Accuracy'])
plt.errorbar(range(len(results_df)), results_df['Mean Accuracy'], 
             yerr=results_df['Std Accuracy'], fmt='o', color='red')
plt.xlabel('Thuật toán')
plt.ylabel('Accuracy')
plt.title('So Sánh Hiệu Suất Các Thuật Toán')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

#### 2.4. Dimensionality Reduction
- [ ] **PCA (Principal Component Analysis)**
  - Giảm chiều dữ liệu
  - Phân tích variance explained
  - Visualization 2D/3D
  
- [ ] **t-SNE**
  - Non-linear dimensionality reduction
  - Visualization clusters
  
- [ ] **LDA (Linear Discriminant Analysis)**
  - Supervised dimensionality reduction
  - Maximize class separability

**Code mẫu:**
```python
# Dimensionality Reduction và Visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(features)

# Cumulative variance explained
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Số thành phần')
plt.ylabel('Tổng phương sai giải thích')
plt.title('PCA - Cumulative Explained Variance')
plt.grid(True)
plt.show()

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_pca)

# Visualize
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Image Features')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(['Healthy', 'Angular Leaf Spot', 'Bean Rust'])
plt.show()
```

#### 2.5. Ensemble Methods - QUAN TRỌNG
- [ ] **Voting Classifier**
  - Hard voting
  - Soft voting
  - Kết hợp nhiều mô hình
  
- [ ] **Bagging**
  - Bootstrap aggregating
  - Reduce variance
  
- [ ] **Boosting**
  - AdaBoost
  - Gradient Boosting
  - XGBoost
  - LightGBM
  
- [ ] **Stacking**
  - Meta-learner
  - Multi-level ensemble

**Code mẫu:**
```python
# Ensemble Methods
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Tạo base classifiers
clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2 = SVC(kernel='rbf', probability=True, random_state=42)
clf3 = KNeighborsClassifier(n_neighbors=5)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('rf', clf1), ('svm', clf2), ('knn', clf3)],
    voting='soft'
)

# AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)

# Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# XGBoost
xgb_clf = XGBClassifier(n_estimators=100, random_state=42)

# So sánh ensemble methods
ensemble_methods = {
    'Voting': voting_clf,
    'AdaBoost': ada_clf,
    'Gradient Boosting': gb_clf,
    'XGBoost': xgb_clf
}

for name, clf in ensemble_methods.items():
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

### PHẦN 3: ĐÁNH GIÁ VÀ TỐI ƯU (20%)

#### 3.1. Cross-Validation
- [ ] **K-Fold Cross-Validation**
- [ ] **Stratified K-Fold**
- [ ] **Leave-One-Out**

#### 3.2. Hyperparameter Tuning
- [ ] **Grid Search**
- [ ] **Random Search**
- [ ] **Bayesian Optimization**

**Code mẫu:**
```python
# Hyperparameter Tuning với Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Định nghĩa parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Grid Search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Test với best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

#### 3.3. Đánh Giá Toàn Diện
- [ ] **Confusion Matrix**
- [ ] **Precision, Recall, F1-Score**
- [ ] **ROC Curve và AUC**
- [ ] **Classification Report**
- [ ] **Learning Curves**
- [ ] **Validation Curves**

**Code mẫu:**
```python
# Đánh giá toàn diện
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 2. Classification Report
print(classification_report(y_test, y_pred, target_names=class_names))

# 3. ROC Curve cho multi-class
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = model.predict_proba(X_test)

plt.figure(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Multi-class')
plt.legend()
plt.show()

# 4. Learning Curves
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

---

### PHẦN 4: BÁO CÁO VÀ TRÌNH BÀY (10%)

#### 4.1. Báo Cáo Kỹ Thuật
- [ ] **Mô tả bài toán và dataset**
- [ ] **Phương pháp tiền xử lý**
- [ ] **Các thuật toán đã sử dụng**
- [ ] **Kết quả thực nghiệm**
- [ ] **Phân tích và so sánh**
- [ ] **Kết luận và hướng phát triển**

#### 4.2. Code và Documentation
- [ ] **Jupyter Notebook đầy đủ**
- [ ] **Comments tiếng Việt**
- [ ] **README.md hướng dẫn**
- [ ] **Requirements.txt**

---

## 📊 CHECKLIST HOÀN THIỆN BÀI TẬP

### TUẦN 1: KHAI PHÁ VÀ PHÂN TÍCH DỮ LIỆU

**Ngày 1-2: Khám phá và làm sạch dữ liệu**
- [x] Tải và load dataset
- [ ] EDA chi tiết với visualizations
- [ ] Phát hiện và xử lý outliers
- [ ] Chuẩn hóa dữ liệu

**Ngày 3-4: Trích xuất đặc trưng**
- [ ] Extract color features (histograms, moments, HSV)
- [ ] Extract texture features (GLCM, LBP, Gabor)
- [ ] Extract shape features
- [ ] CNN features extraction

**Ngày 5-7: Áp dụng thuật toán Data Mining**
- [ ] K-Means clustering với phân tích
- [ ] Hierarchical clustering
- [ ] DBSCAN
- [ ] Association rules mining
- [ ] PCA và t-SNE visualization

### TUẦN 2: MÔ HÌNH HÓA VÀ ĐÁNH GIÁ

**Ngày 8-10: Classification với nhiều thuật toán**
- [ ] Decision Tree + visualization
- [ ] Random Forest + feature importance
- [ ] SVM (linear và RBF kernel)
- [ ] Naive Bayes
- [ ] k-NN
- [ ] Neural Networks (MLP)
- [ ] CNN và Transfer Learning

**Ngày 11-12: Ensemble và Optimization**
- [ ] Voting Classifier
- [ ] Bagging
- [ ] Boosting (AdaBoost, GradientBoosting, XGBoost)
- [ ] Stacking
- [ ] Hyperparameter tuning
- [ ] Cross-validation

**Ngày 13-14: Đánh giá và báo cáo**
- [ ] Comprehensive evaluation
- [ ] Comparison tables và charts
- [ ] Viết báo cáo kỹ thuật
- [ ] Chuẩn bị presentation
- [ ] Code cleanup và documentation

---

## 📚 TÀI LIỆU THAM KHẢO TỪ NHIỀU NGUỒN

### 🔗 Kaggle Notebooks (Top Contributors)
1. **Leaf Disease Detection w/Hybrid Model (ViT, SVM)** - arturo-bandini-jr
   - Hybrid approach: Vision Transformer + SVM
   - Accuracy: 98%+
   - Link: https://www.kaggle.com/code/banddaniel/leaf-disease-detection-w-hybrid-model-vit-svm

2. **Bean Leaf Lesions Pytorch ACC 99%** - Chihjung Wang
   - PyTorch implementation
   - 99% accuracy
   - Link: https://www.kaggle.com/code/chihjungwang/bean-leaf-lesions-pytorch-acc-99

3. **Bean Leaf Lesions Classification** - Akhil Chhibber
   - Complete pipeline
   - Link: https://www.kaggle.com/code/akhilchibber/bean-leaf-lesions-classification

### 🔗 GitHub Repositories
1. **akhilchibber/Bean-Leaf-Lesions-Classification**
   - Complete implementation
   - MIT License
   - https://github.com/akhilchibber/Bean-Leaf-Lesions-Classification

### 🔗 Similar Datasets cho Comparison
1. **Rice Leaf Dataset** - 3355 files, 8 GB
2. **Maize Leaf** - 8852 files, 127 MB
3. **Corn/Maize Leaf Disease** - 4188 files, 169 MB
4. **Leaf Type Detection** - 4660 files, 48 MB

### 📖 Papers và Techniques
1. **K-Means Clustering for Image Segmentation**
   - Lloyd's algorithm
   - Elbow method cho optimal k

2. **Association Rule Mining for Feature Analysis**
   - Apriori algorithm
   - FP-Growth

3. **Ensemble Methods**
   - Bagging, Boosting, Stacking
   - Voting classifiers

4. **Transfer Learning**
   - ResNet, EfficientNet, VGG
   - Fine-tuning strategies

---

## 💻 CÀI ĐẶT MÔI TRƯỜNG

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt packages
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scikit-image
pip install opencv-python pillow
pip install torch torchvision
pip install xgboost lightgbm
pip install mlxtend  # Cho association rules
pip install yellowbrick  # Cho visualizations
pip install shap  # Cho model interpretation
pip install jupyter notebook
```

---

## 🎯 TIÊU CHÍ ĐÁNH GIÁ BÀI TẬP LỚN

| Tiêu chí | Tỷ trọng | Yêu cầu |
|----------|----------|---------|
| **Thu thập & Tiền xử lý** | 20% | - Load data đúng<br>- EDA chi tiết<br>- Feature engineering |
| **Thuật toán Data Mining** | 40% | - Clustering (K-Means, Hierarchical)<br>- Association rules<br>- Classification (>=5 thuật toán)<br>- Ensemble methods |
| **Đánh giá & So sánh** | 20% | - Cross-validation<br>- Multiple metrics<br>- Comparison tables<br>- Visualization |
| **Báo cáo & Code** | 20% | - Báo cáo rõ ràng<br>- Code sạch, có comments<br>- Documentation đầy đủ<br>- Presentation |

---

## 🚀 HƯỚNG PHÁT TRIỂN THÊM (Điểm Cộng)

1. **Web Application**
   - Flask/Django web app
   - Upload ảnh và predict
   - REST API

2. **Mobile App**
   - TensorFlow Lite
   - Deploy trên Android/iOS

3. **Advanced Techniques**
   - AutoML
   - Neural Architecture Search
   - Explainable AI (SHAP, LIME)

4. **Real-world Testing**
   - Test với ảnh thực tế
   - A/B testing
   - User feedback

---

## 📝 MẪU BÁO CÁO

### Cấu trúc báo cáo (15-20 trang):

1. **Trang bìa** (1 trang)
2. **Mục lục** (1 trang)
3. **Giới thiệu** (1-2 trang)
   - Bối cảnh bài toán
   - Mục tiêu nghiên cứu
   - Phạm vi và giới hạn

4. **Dữ liệu và Tiền xử lý** (3-4 trang)
   - Mô tả dataset
   - EDA với charts
   - Feature engineering
   - Data cleaning

5. **Phương pháp** (5-6 trang)
   - Clustering analysis
   - Association rules
   - Classification algorithms
   - Ensemble methods

6. **Kết quả** (3-4 trang)
   - Performance metrics
   - Comparison tables
   - Confusion matrices
   - Learning curves

7. **Thảo luận** (1-2 trang)
   - Phân tích kết quả
   - So sánh thuật toán
   - Challenges và solutions

8. **Kết luận** (1 trang)
   - Tóm tắt
   - Đóng góp
   - Hướng phát triển

9. **Tài liệu tham khảo** (1 trang)

---

## 🎓 KẾT LUẬN

Bài tập lớn này yêu cầu:
- ✅ Hiểu sâu về Data Mining
- ✅ Thực hành nhiều thuật toán
- ✅ Phân tích và đánh giá toàn diện
- ✅ Báo cáo chuyên nghiệp

**Thời gian 2 tuần là VỪA ĐỦ** nếu bạn:
- Làm việc có kế hoạch
- Chia nhỏ công việc theo ngày
- Tham khảo nhiều nguồn
- Code và test liên tục

**Chúc bạn làm bài tập thành công! 🎉**

---

**Liên hệ hỗ trợ:**
- GitHub Issues: [Open Issue](https://github.com/akhilchibber/Bean-Leaf-Lesions-Classification/issues)
- Email: support@example.com
