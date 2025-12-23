# 🚀 HƯỚNG DẪN CHẠY NOTEBOOK

## ⚠️ QUAN TRỌNG: Notebook này được thiết kế cho **GOOGLE COLAB**

---

## 📋 **CHUẨN BỊ:**

### 1. Upload dữ liệu lên Google Drive
- Tạo thư mục: `Bean-Leaf-Lesions-Classification/INPUT_DATASET/`
- Upload 2 thư mục:
  - `train/` (1,034 ảnh)
  - `val/` (133 ảnh)

### 2. Cấu trúc thư mục trên Google Drive:
```
MyDrive/
└── Bean-Leaf-Lesions-Classification/
    └── INPUT_DATASET/
        ├── train/
        │   ├── healthy/
        │   ├── angular_leaf_spot/
        │   └── bean_rust/
        └── val/
            ├── healthy/
            ├── angular_leaf_spot/
            └── bean_rust/
```

---

## 🏃 **CHẠY NOTEBOOK:**

### **Bước 1: Mở Google Colab**
1. Truy cập: https://colab.research.google.com/
2. Click `File` → `Upload notebook`
3. Chọn file `Bean-Leaf-Lesions.ipynb`

### **Bước 2: Chạy từng cell theo thứ tự**

#### **✅ Cell 1-2: Markdown** 
- Bỏ qua (chỉ là text)

#### **✅ Cell 3: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Chạy và cho phép quyền truy cập Google Drive
- Nhập mã xác thực khi được yêu cầu

#### **✅ Cell 4: Import thư viện cơ bản**
```python
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from glob import glob
```
- Chạy trực tiếp, không có lỗi

#### **✅ Cell 7-8: Đường dẫn dữ liệu**
```python
train_df = pd.DataFrame({"path":[], "label":[], "class_id":[]})
val_df = pd.DataFrame({"path":[], "label":[], "class_id":[]})

train_path = '/content/drive/MyDrive/Bean-Leaf-Lesions-Classification/INPUT_DATASET/train'
val_path = '/content/drive/MyDrive/Bean-Leaf-Lesions-Classification/INPUT_DATASET/val'
```
- ⚠️ **Kiểm tra đường dẫn khớp với thư mục trên Drive của bạn**

#### **✅ Cell 9-11: Load dữ liệu**
- Chạy các cell để load ảnh và tạo DataFrame
- Kiểm tra output: `✅ Đã tải X ảnh huấn luyện và Y ảnh kiểm định`

---

## 🔧 **XỬ LÝ LỖI THƯỜNG GẶP:**

### **1. Lỗi: "No such file or directory"**
**Nguyên nhân:** Đường dẫn không đúng

**Cách fix:**
```python
# Kiểm tra đường dẫn
!ls /content/drive/MyDrive/

# Nếu thư mục khác, sửa lại đường dẫn:
train_path = '/content/drive/MyDrive/[TÊN_THƯ_MỤC_CỦA_BẠN]/INPUT_DATASET/train'
```

### **2. Lỗi: "ModuleNotFoundError"**
**Nguyên nhân:** Thiếu thư viện

**Cách fix:**
```python
!pip install [tên_thư_viện]
```

Ví dụ:
- `!pip install opencv-python`
- `!pip install scikit-image`
- `!pip install xgboost`
- `!pip install lightgbm`

### **3. Lỗi: "CUDA out of memory"**
**Nguyên nhân:** GPU không đủ bộ nhớ

**Cách fix:**
- Runtime → Change runtime type → GPU → T4 GPU
- Giảm BATCH_SIZE từ 32 xuống 16 hoặc 8

### **4. Lỗi Data Mining cells (Feature Extraction)**
**Cell 26-28:** Có thể mất 10-20 phút để chạy

**Lưu ý:**
```python
# Nếu đã chạy rồi, có thể load lại features đã lưu:
X_features = np.load('features.npy')
y_labels = np.load('labels.npy')
```

---

## 📊 **THEO DÕI TIẾN TRÌNH:**

### **Phần 1: EDA (Cell 1-24)**
- Thời gian: ~2-3 phút
- Output: Biểu đồ phân phối, sample images

### **Phần 2: Data Mining (Cell 25-42)**
- **Feature Extraction:** ~15-20 phút
- **Clustering:** ~5-10 phút
- **Classification:** ~3-5 phút
- **Ensemble:** ~5-10 phút

### **Phần 3: Deep Learning (Cell 58-72)**
- **Data Augmentation:** ~2 phút
- **Model Training:** ~30-60 phút (tùy epochs)

### **Phần 4: Evaluation (Cell 73-126)**
- **Metrics & Visualization:** ~3-5 phút

---

## 💾 **LƯU KẾT QUẢ:**

### **Tự động lưu trong notebook:**
- Features: `features.npy`, `labels.npy`
- Models: `best_bean_leaf_model.pth`
- Results: DataFrame trong biến

### **Lưu thủ công:**
```python
# Lưu model
torch.save(model.state_dict(), '/content/drive/MyDrive/model.pth')

# Lưu results
results_df.to_csv('/content/drive/MyDrive/results.csv', index=False)
```

---

## ⏱️ **THỜI GIAN CHẠY DỰ KIẾN:**

| Phần | Thời gian | Ghi chú |
|------|-----------|---------|
| **EDA** | 2-3 phút | Chạy nhanh |
| **Feature Extraction** | 15-20 phút | Chạy 1 lần, lưu lại |
| **Clustering** | 5-10 phút | OK |
| **Classification** | 3-5 phút | OK |
| **Ensemble** | 5-10 phút | OK |
| **Deep Learning** | 30-60 phút | Tùy epochs |
| **Evaluation** | 3-5 phút | OK |
| **TỔNG** | ~1-2 giờ | Chạy đầy đủ |

---

## 🎯 **CHECKLIST TRƯỚC KHI NỘP BÀI:**

- [ ] Đã chạy đầy đủ tất cả các cell
- [ ] Không có cell nào báo lỗi
- [ ] Các biểu đồ hiển thị đầy đủ
- [ ] Kết quả Classification + Ensemble đã có
- [ ] Ma trận nhầm lẫn và ROC curves đã hiển thị
- [ ] Feature importance đã phân tích
- [ ] Có phần tổng kết và kết luận

---

## 📝 **GHI CHÚ:**

1. **Khuyến khích:** Chạy trên Colab với GPU T4 (miễn phí)
2. **Thời gian:** Dự trù 2-3 giờ để chạy toàn bộ
3. **Lưu ý:** Feature Extraction chỉ cần chạy 1 lần, sau đó load lại
4. **Quan trọng:** Lưu notebook thường xuyên (Ctrl+S)

---

## 🆘 **HỖ TRỢ:**

Nếu gặp lỗi không fix được:
1. Kiểm tra lại đường dẫn dữ liệu
2. Đảm bảo đã mount Google Drive
3. Kiểm tra log lỗi để biết thiếu thư viện nào
4. Restart runtime và chạy lại từ đầu

---

**🎉 CHÚC BẠN CHẠY THÀNH CÔNG!**
