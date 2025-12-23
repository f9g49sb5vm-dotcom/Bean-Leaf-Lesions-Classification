# ⚠️ TROUBLESHOOTING - XỬ LÝ LỖI THƯỜNG GẶP

## 🔴 **LỖI ĐÃ FIX:**

### ✅ **1. Lỗi Cell 3: Google Drive Mount**
**Lỗi trước đây:**
```python
from google.colab import drivedrive.mount('Bean-Leaf-Lesions-Classification.lnk')
```

**Đã sửa thành:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Giải thích:** Thiếu xuống dòng và đường dẫn sai.

---

### ✅ **2. Lỗi Import Trùng Lặp**
**Vấn đề:** Nhiều cell import `torch`, `nn`, `models` lặp lại

**Giải pháp:** Đã tập trung tất cả import vào Cell 5 (cell import chính)
- ✅ Cell 5: Import tất cả thư viện (pandas, numpy, torch, sklearn, cv2, etc.)
- ✅ Các cell khác: Không import lại nữa

---

### ✅ **3. Lỗi DataFrame Format**
**Lỗi trước đây:**
```python
# - label: tên lớp (dạng text)val_df = pd.DataFrame(...)
# - class_id: ID số...train_df = pd.DataFrame(...)
```

**Đã sửa thành:**
```python
# - label: tên lớp (dạng text)
# - class_id: ID số của lớp (0, 1, 2)

train_df = pd.DataFrame({"path":[], "label":[], "class_id":[]})
val_df = pd.DataFrame({"path":[], "label":[], "class_id":[]})
```

---

## 🟡 **LỖI CÓ THỂ GẶP KHI CHẠY:**

### **1. ModuleNotFoundError: No module named 'cv2'**

**Nguyên nhân:** Chưa cài OpenCV

**Cách fix:**
```python
!pip install opencv-python
```

Hoặc chạy Cell 2 (Auto-install cell) để tự động cài.

---

### **2. FileNotFoundError: [Errno 2] No such file or directory**

**Nguyên nhân:** Chưa upload dữ liệu lên Google Drive hoặc đường dẫn sai

**Cách fix:**

**Bước 1: Kiểm tra đường dẫn**
```python
!ls /content/drive/MyDrive/
```

**Bước 2: Xác nhận cấu trúc thư mục**
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

**Bước 3: Sửa đường dẫn nếu cần**
```python
# Sửa ở Cell 8
train_path = '/content/drive/MyDrive/[TÊN_THƯ_MỤC_CỦA_BẠN]/INPUT_DATASET/train'
val_path = '/content/drive/MyDrive/[TÊN_THƯ_MỤC_CỦA_BẠN]/INPUT_DATASET/val'
```

---

### **3. RuntimeError: CUDA out of memory**

**Nguyên nhân:** GPU không đủ bộ nhớ

**Cách fix:**

**Option 1: Giảm BATCH_SIZE**
```python
# Sửa ở Cell 63
BATCH_SIZE = 16  # Hoặc 8 thay vì 32
```

**Option 2: Dùng GPU mạnh hơn**
- Runtime → Change runtime type → Hardware accelerator → GPU → T4 GPU

**Option 3: Clear cache**
```python
import torch
torch.cuda.empty_cache()
```

---

### **4. ModuleNotFoundError: No module named 'xgboost'**

**Nguyên nhân:** Chưa cài XGBoost (dùng cho Ensemble Methods)

**Cách fix:**
```python
!pip install xgboost lightgbm
```

**Lưu ý:** Phần này không bắt buộc. Nếu không cần Ensemble Methods, có thể bỏ qua các cell 36-37 (XGBoost, LightGBM).

---

### **5. UserWarning: Palette images with Transparency**

**Nguyên nhân:** Một số ảnh có format palette với alpha channel

**Cách fix:** Đã xử lý tự động trong code
```python
img = Image.open(image_path).convert("RGB")  # Convert về RGB
```

Nếu vẫn gặp lỗi, thêm:
```python
import warnings
warnings.filterwarnings('ignore')
```

---

### **6. ValueError: too many values to unpack**

**Nguyên nhân:** Dữ liệu không đúng format hoặc thiếu ảnh

**Cách fix:**

**Kiểm tra số lượng ảnh:**
```python
print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"\nClasses distribution:")
print(train_df['label'].value_counts())
```

**Đảm bảo có ít nhất:**
- Train: > 100 ảnh
- Val: > 30 ảnh
- Mỗi class có ít nhất 10 ảnh

---

### **7. Notebook bị disconnect sau vài giờ**

**Nguyên nhân:** Google Colab free có giới hạn thời gian (12h max)

**Cách fix:**

**Option 1: Lưu checkpoint thường xuyên**
```python
# Sau mỗi epoch quan trọng
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, '/content/drive/MyDrive/checkpoint.pth')
```

**Option 2: Giảm epochs**
```python
NUM_EPOCHS = 10  # Thay vì 20 hoặc 30
```

**Option 3: Chạy ở Google Colab Pro**
- Unlimited runtime
- Faster GPUs (A100, V100)

---

## 🟢 **TIPS CHẠY NHANH:**

### **1. Bỏ qua Feature Extraction nếu đã chạy**

Nếu đã chạy Cell 28 (extract features) một lần, có thể lưu lại và load:

```python
# Sau khi extract xong, lưu lại:
np.save('/content/drive/MyDrive/X_train_features.npy', X_train_scaled)
np.save('/content/drive/MyDrive/y_train.npy', y_train)
np.save('/content/drive/MyDrive/X_val_features.npy', X_val_scaled)
np.save('/content/drive/MyDrive/y_val.npy', y_val)

# Lần sau chỉ cần load:
X_train_scaled = np.load('/content/drive/MyDrive/X_train_features.npy')
y_train = np.load('/content/drive/MyDrive/y_train.npy')
X_val_scaled = np.load('/content/drive/MyDrive/X_val_features.npy')
y_val = np.load('/content/drive/MyDrive/y_val.npy')
```

**Tiết kiệm:** ~15-20 phút mỗi lần chạy lại!

---

### **2. Chạy với ít epochs để test**

Trước khi chạy đầy đủ, test với ít epochs:

```python
NUM_EPOCHS = 2  # Thay vì 20
```

Xác nhận không lỗi → Tăng lên số epochs thật.

---

### **3. Sử dụng Model nhỏ hơn**

Nếu GPU yếu, dùng EfficientNet thay vì ResNet:

```python
model = create_model('efficientnet_b0')  # 5M params thay vì 25M
```

---

### **4. Tắt Data Augmentation khi test**

```python
# Tạm thời dùng val_transforms cho cả train
train_loader = DataLoader(
    MyDataset(train_df, val_transforms),  # Dùng val_transforms
    batch_size=BATCH_SIZE,
    shuffle=True
)
```

Chạy nhanh hơn 2-3 lần!

---

## 📊 **KIỂM TRA TRƯỚC KHI NỘP BÀI:**

- [ ] Cell 3: Mount Drive thành công
- [ ] Cell 9-11: Load được dữ liệu (hiển thị số ảnh)
- [ ] Cell 13-24: EDA charts hiển thị đầy đủ
- [ ] Cell 28: Feature extraction hoàn thành (có shape output)
- [ ] Cell 30-32: Clustering chạy được (có Silhouette score)
- [ ] Cell 33-35: Classification có accuracy > 70%
- [ ] Cell 36-37: Ensemble methods chạy được
- [ ] Cell 69: Model training hoàn thành
- [ ] Cell 76+: Evaluation metrics và confusion matrix hiển thị
- [ ] Cell cuối: Có kết luận và tổng kết

---

## 🆘 **LIÊN HỆ HỖ TRỢ:**

Nếu vẫn gặp lỗi không fix được:

1. **Screenshot lỗi** (toàn bộ error message)
2. **Note cell nào** đang chạy
3. **Kiểm tra:** Runtime type (CPU/GPU/TPU)
4. **Restart runtime** và thử lại

**Thường 90% lỗi do:**
- Chưa mount Drive
- Đường dẫn dữ liệu sai
- Chưa cài thư viện

---

**🎯 CHÚC BẠN CHẠY THÀNH CÔNG!**
