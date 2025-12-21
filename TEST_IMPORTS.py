"""
Script kiểm tra nhanh tất cả imports cần thiết cho notebook
Chạy script này để kiểm tra xem đã cài đủ thư viện chưa
"""

print("=" * 60)
print("KIỂM TRA THƯ VIỆN CHO BEAN LEAF LESIONS CLASSIFICATION")
print("=" * 60)

# 1. Thư viện cơ bản
print("\n[1/5] Kiểm tra thư viện cơ bản...")
try:
    import pandas as pd
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import os
    from glob import glob
    print("✅ Thư viện cơ bản: OK")
except ImportError as e:
    print(f"❌ Lỗi: {e}")
    print("   Chạy: pip install pandas numpy pillow matplotlib")

# 2. PyTorch
print("\n[2/5] Kiểm tra PyTorch...")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    import torchvision.transforms as transforms
    from torchvision import models
    print(f"✅ PyTorch: OK (version {torch.__version__})")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ Lỗi: {e}")
    print("   Chạy: pip install torch torchvision")

# 3. scikit-learn
print("\n[3/5] Kiểm tra scikit-learn...")
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, 
                                 BaggingClassifier, AdaBoostClassifier, 
                                 GradientBoostingClassifier)
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, confusion_matrix, classification_report,
                                silhouette_score)
    from sklearn.model_selection import train_test_split
    print("✅ scikit-learn: OK")
except ImportError as e:
    print(f"❌ Lỗi: {e}")
    print("   Chạy: pip install scikit-learn")

# 4. OpenCV & skimage (cho feature extraction)
print("\n[4/5] Kiểm tra OpenCV & scikit-image...")
try:
    import cv2
    from skimage import color, feature
    from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
    from scipy import ndimage
    print("✅ OpenCV & scikit-image: OK")
except ImportError as e:
    print(f"❌ Lỗi: {e}")
    print("   Chạy: pip install opencv-python scikit-image scipy")

# 5. XGBoost & LightGBM (cho ensemble)
print("\n[5/5] Kiểm tra XGBoost & LightGBM...")
try:
    import xgboost as xgb
    import lightgbm as lgb
    print("✅ XGBoost & LightGBM: OK")
except ImportError as e:
    print(f"⚠️  Cảnh báo: {e}")
    print("   (Không bắt buộc) Chạy: pip install xgboost lightgbm")

print("\n" + "=" * 60)
print("KẾT QUẢ KIỂM TRA")
print("=" * 60)
print("\n✅ Nếu tất cả đều OK, bạn có thể chạy notebook!")
print("❌ Nếu có lỗi, cài thư viện theo hướng dẫn phía trên\n")
print("=" * 60)
