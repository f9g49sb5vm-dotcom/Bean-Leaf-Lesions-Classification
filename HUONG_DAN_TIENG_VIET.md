# 📚 HƯỚNG DẪN DỊCH VÀ GHI CHÚ - BEAN LEAF LESIONS CLASSIFICATION

## 📖 Tổng Quan
Tài liệu này cung cấp bản dịch tiếng Việt và ghi chú giải thích chi tiết cho toàn bộ notebook phân loại tổn thương lá đậu.

---

## 🔤 BẢNG TỪ VỰNG CHUYÊN NGÀNH

### Thuật ngữ Machine Learning
| English | Tiếng Việt | Giải thích |
|---------|------------|------------|
| Deep Learning | Học sâu | Học máy sử dụng mạng nơ-ron nhiều lớp |
| Transfer Learning | Học chuyển giao | Sử dụng lại kiến thức từ mô hình đã huấn luyện |
| Overfitting | Quá khớp | Mô hình học quá tốt trên dữ liệu train, kém trên dữ liệu mới |
| Data Augmentation | Tăng cường dữ liệu | Tạo thêm dữ liệu huấn luyện bằng các phép biến đổi |
| Epoch | Epoch | Một lần duyệt qua toàn bộ tập dữ liệu huấn luyện |
| Batch | Lô/Batch | Nhóm mẫu xử lý cùng lúc |
| Loss Function | Hàm mất mát | Đo lường sai số của mô hình |
| Optimizer | Bộ tối ưu | Thuật toán cập nhật trọng số mô hình |
| Learning Rate | Tỷ lệ học | Tốc độ cập nhật trọng số |
| Validation | Kiểm định | Đánh giá mô hình trên dữ liệu chưa thấy |
| Accuracy | Độ chính xác | Tỷ lệ dự đoán đúng |
| Precision | Độ chính xác (dương tính) | Trong các mẫu dự đoán dương, có bao nhiêu đúng |
| Recall | Độ thu hồi | Trong các mẫu thực sự dương, phát hiện được bao nhiêu |
| F1-Score | Điểm F1 | Trung bình điều hòa của Precision và Recall |
| Confusion Matrix | Ma trận nhầm lẫn | Bảng thống kê dự đoán đúng/sai |

### Thuật ngữ Computer Vision
| English | Tiếng Việt | Giải thích |
|---------|------------|------------|
| CNN | Mạng nơ-ron tích chập | Kiến trúc mạng chuyên xử lý ảnh |
| Convolution | Tích chập | Phép toán trích xuất đặc trưng từ ảnh |
| Pooling | Gộp | Giảm kích thước ảnh giữ lại thông tin quan trọng |
| Feature Map | Bản đồ đặc trưng | Kết quả sau mỗi lớp tích chập |
| Dropout | Dropout | Kỹ thuật tắt ngẫu nhiên nơ-ron để giảm overfitting |
| Normalization | Chuẩn hóa | Đưa giá trị về phạm vi chuẩn |
| ResNet | ResNet | Kiến trúc sử dụng kết nối tàn dư |
| EfficientNet | EfficientNet | Kiến trúc tối ưu hiệu quả |

### Thuật ngữ Nông nghiệp
| English | Tiếng Việt | 
|---------|------------|
| Bean Leaf | Lá đậu |
| Lesion | Tổn thương/Vết bệnh |
| Healthy | Khỏe mạnh |
| Angular Leaf Spot | Đốm góc lá |
| Bean Rust | Gỉ sắt đậu |
| Disease Detection | Phát hiện bệnh |
| Crop Monitoring | Giám sát cây trồng |

---

## 📝 GHI CHÚ CHI TIẾT THEO BƯỚC

### BƯỚC 1: CHUẨN BỊ DỮ LIỆU

#### 1.1 Nhập Thư Viện
**Giải thích:**
- **pandas:** Xử lý dữ liệu dạng bảng (DataFrame). Rất mạnh cho việc lọc, sắp xếp, thống kê dữ liệu.
- **numpy:** Thư viện tính toán khoa học. Xử lý mảng nhiều chiều, các phép toán toán học phức tạp.
- **PIL (Python Imaging Library):** Mở, lưu, và xử lý ảnh nhiều định dạng (JPG, PNG, etc.)
- **matplotlib:** Vẽ biểu đồ, trực quan hóa dữ liệu. Tương tự MATLAB.
- **os:** Tương tác với hệ điều hành (tạo/xóa thư mục, đường dẫn file)
- **glob:** Tìm kiếm file theo pattern (ví dụ: tất cả file .jpg)

#### 1.2 Tải và Tổ Chức Dữ Liệu
**Cấu trúc thư mục:**
```
INPUT_DATASET/
├── train/                 # Dữ liệu huấn luyện
│   ├── healthy/          # Lá khỏe mạnh
│   ├── angular_leaf_spot/ # Lá bị đốm góc
│   └── bean_rust/        # Lá bị gỉ sắt
└── val/                  # Dữ liệu kiểm định
    ├── healthy/
    ├── angular_leaf_spot/
    └── bean_rust/
```

**Hàm load_images:**
- Duyệt qua từng thư mục con (mỗi thư mục = 1 lớp)
- Tìm tất cả file .jpg trong thư mục
- Lưu đường dẫn + nhãn vào DataFrame
- Trả về DataFrame với 3 cột: path, label, class_id

**Tại sao cần DataFrame?**
- Dễ quản lý và truy vấn dữ liệu
- Hỗ trợ filter, group, statistic
- Tích hợp tốt với PyTorch DataLoader

#### 1.3 Trực Quan Hóa Dữ Liệu
**Mục đích:**
- Kiểm tra dữ liệu đã load đúng chưa
- Quan sát đặc điểm của từng lớp
- Phát hiện ảnh lỗi, ảnh nhiễu

---

### BƯỚC 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)

#### 2.1 Thống Kê Bộ Dữ Liệu
**Các chỉ số quan trọng:**
- **Tổng số ảnh:** 1,167 ảnh (1,034 train + 133 val)
- **Số lớp:** 3 lớp
- **Tỷ lệ train/val:** ~88%/12%

**Tại sao cần thống kê?**
- Hiểu quy mô dữ liệu
- Đánh giá đủ dữ liệu để train không
- Xác định cách chia train/val có hợp lý không

#### 2.2 Phân Tích Phân Phối Lớp
**Kiểm tra cân bằng dữ liệu:**
- Lớp nào có nhiều mẫu nhất?
- Có lớp nào thiếu dữ liệu nghiêm trọng không?
- Cần áp dụng kỹ thuật cân bằng không? (oversampling, undersampling, class weights)

**Vấn đề mất cân bằng lớp:**
- Nếu 1 lớp có 90% dữ liệu, mô hình sẽ bias về lớp đó
- Mô hình dự đoán tất cả là lớp đa số → Accuracy cao nhưng vô dụng
- Giải pháp: Tăng trọng số cho lớp thiểu số, augmentation, focal loss

#### 2.3 Phân Tích Kích Thước Ảnh
**Thống kê:**
- Width (chiều rộng): Min, Max, Mean
- Height (chiều cao): Min, Max, Mean  
- Aspect Ratio (tỷ lệ khung hình): W/H

**Tại sao quan trọng?**
- Xác định kích thước resize phù hợp
- Phát hiện ảnh bị méo, ảnh lỗi
- Quyết định có cần padding không

---

### BƯỚC 3: TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU

#### 3.1 Các Kỹ Thuật Tăng Cường
**1. Resize & Crop:**
- **Resize(256, 256):** Đưa ảnh về kích thước chuẩn
- **RandomResizedCrop(224):** Crop ngẫu nhiên vùng ảnh → tạo biến thể
- **Lợi ích:** Mô hình học nhận diện vật thể ở các vị trí khác nhau

**2. Flip (Lật):**
- **HorizontalFlip:** Lật ngang (trái ↔ phải)
- **VerticalFlip:** Lật dọc (trên ↔ dưới)
- **Lý do:** Lá cây có thể xuất hiện ở bất kỳ hướng nào

**3. Rotation (Xoay):**
- **RandomRotation(±30°):** Xoay ngẫu nhiên trong khoảng ±30 độ
- **Lợi ích:** Mô hình robust với các góc chụp khác nhau

**4. Color Jitter (Thay đổi màu):**
- **Brightness:** Độ sáng (±20%)
- **Contrast:** Độ tương phản (±20%)
- **Saturation:** Độ bão hòa màu (±20%)
- **Hue:** Sắc độ màu (±10%)
- **Lý do:** Điều kiện ánh sáng khác nhau khi chụp

**5. Normalization (Chuẩn hóa):**
- **ImageNet statistics:** mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Tại sao dùng ImageNet?** Mô hình pretrained được train trên ImageNet đã chuẩn hóa như vậy
- **Công thức:** normalized = (pixel - mean) / std

#### 3.2 Tầm Quan Trọng của Augmentation
**Không có Augmentation:**
- Mô hình chỉ thấy 1,034 ảnh
- Dễ overfitting (học thuộc lòng)
- Kém với dữ liệu mới

**Có Augmentation:**
- Mô hình thấy hàng triệu biến thể
- Học các đặc trưng tổng quát
- Robust với nhiễu, góc chụp, ánh sáng

---

### BƯỚC 4: KIẾN TRÚC MÔ HÌNH

#### 4.1 So Sánh Các Kiến Trúc

**ResNet50:**
- **Ý tưởng:** Kết nối tàn dư (skip connections) giải quyết vấn đề vanishing gradient
- **Công thức:** F(x) + x thay vì chỉ F(x)
- **Ưu điểm:** Train được mạng rất sâu (50-152 layers)
- **Khi nào dùng:** Cân bằng tốt giữa accuracy và tốc độ

**EfficientNetB0:**
- **Ý tưởng:** Compound scaling (tăng đồng thời depth, width, resolution)
- **Ưu điểm:** Ít tham số nhất (5M) nhưng accuracy cao
- **Khi nào dùng:** Triển khai mobile, embedded, edge devices

**VGG16:**
- **Ý tưởng:** Mạng sâu đơn giản, chỉ dùng Conv 3x3
- **Nhược điểm:** Rất nặng (138M tham số)
- **Ưu điểm:** Dễ hiểu, dễ implement, accuracy tốt
- **Khi nào dùng:** Research, học tập, có tài nguyên dồi dào

**DenseNet121:**
- **Ý tưởng:** Mỗi layer kết nối với tất cả layers trước đó
- **Ưu điểm:** Gradient flow tốt, ít tham số hơn ResNet
- **Khi nào dùng:** Cần accuracy cao với ít tham số

#### 4.2 Transfer Learning
**Nguyên lý:**
1. **Pretrained:** Mô hình đã học từ ImageNet (14M ảnh, 1000 lớp)
2. **Feature Extractor:** Các lớp đầu học các đặc trưng cơ bản (edges, textures, patterns)
3. **Fine-tuning:** Chỉ train lại lớp cuối cùng cho bài toán mới

**Tại sao hiệu quả?**
- Đặc trưng low-level (cạnh, góc, màu) là universal
- Tiết kiệm thời gian train (giờ → phút)
- Cần ít dữ liệu hơn (triệu → nghìn ảnh)
- Accuracy cao hơn train from scratch

**Freeze vs Fine-tune:**
- **Freeze backbone:** Giữ nguyên trọng số các lớp đầu, chỉ train lớp cuối
- **Fine-tune:** Train cả backbone với learning rate thấp

---

### BƯỚC 5: HUẤN LUYỆN MÔ HÌNH

#### 5.1 Loss Function & Optimizer

**CrossEntropyLoss:**
- **Dùng cho:** Phân loại multi-class
- **Công thức:** -log(p_correct_class)
- **Giải thích:** Penalize mạnh khi dự đoán sai với confidence cao

**AdamW Optimizer:**
- **Adam:** Adaptive Moment Estimation
- **W (Weight Decay):** Regularization để giảm overfitting
- **Learning Rate:** 0.001 (giảm dần trong quá trình train)
- **Ưu điểm:** Converge nhanh, ít nhạy cảm với hyperparameters

#### 5.2 Learning Rate Scheduling

**ReduceLROnPlateau:**
- **Ý tưởng:** Giảm LR khi validation loss không giảm
- **Patience=3:** Chờ 3 epochs không cải thiện thì giảm LR
- **Factor=0.5:** Giảm LR xuống một nửa
- **Lợi ích:** Tránh stuck ở local minima, fine-tune tốt hơn

#### 5.3 Early Stopping
**Mục đích:** Dừng train khi mô hình bắt đầu overfit

**Cách hoạt động:**
- Monitor validation accuracy
- Lưu best model (accuracy cao nhất)
- Nếu không cải thiện sau `patience` epochs → stop
- Restore best model

**Tại sao quan trọng?**
- Tiết kiệm thời gian (không train vô ích)
- Tránh overfit (dừng đúng lúc)
- Đảm bảo model tốt nhất

---

### BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH

#### 6.1 Ma Trận Nhầm Lẫn (Confusion Matrix)
```
              Dự đoán
            H    A    B
Thực tế H  TP   FN   FN
        A  FP   TP   FN
        B  FP   FN   TP
```

**Giải thích:**
- **TP (True Positive):** Dự đoán đúng
- **FP (False Positive):** Dự đoán dương nhưng thực tế âm
- **FN (False Negative):** Dự đoán âm nhưng thực tế dương

**Ý nghĩa:**
- Đường chéo: Số mẫu dự đoán đúng
- Ngoài đường chéo: Nhầm lẫn giữa các lớp
- Lớp nào bị nhầm nhiều nhất?

#### 6.2 Các Metric

**Accuracy:**
- **Công thức:** (TP + TN) / Total
- **Ý nghĩa:** Tỷ lệ dự đoán đúng tổng thể
- **Hạn chế:** Không phù hợp với dữ liệu mất cân bằng

**Precision:**
- **Công thức:** TP / (TP + FP)
- **Ý nghĩa:** Trong các mẫu dự đoán là dương, có bao nhiêu đúng?
- **Quan trọng khi:** Cost of False Positive cao

**Recall:**
- **Công thức:** TP / (TP + FN)
- **Ý nghĩa:** Trong các mẫu thực sự dương, phát hiện được bao nhiêu?
- **Quan trọng khi:** Cost of False Negative cao

**F1-Score:**
- **Công thức:** 2 * (Precision * Recall) / (Precision + Recall)
- **Ý nghĩa:** Trung bình điều hòa của Precision và Recall
- **Khi nào dùng:** Cân bằng giữa Precision và Recall

---

## 💡 TIPS & BEST PRACTICES

### 1. Chuẩn Bị Dữ Liệu
✅ **Nên:**
- Kiểm tra kỹ dữ liệu trước khi train
- Cân bằng các lớp nếu chênh lệch quá lớn
- Augmentation mạnh mẽ cho dữ liệu nhỏ
- Chia train/val/test rõ ràng

❌ **Không nên:**
- Train trên tất cả dữ liệu (không có validation)
- Bỏ qua outliers và ảnh lỗi
- Augmentation quá mạnh làm mất đặc trưng

### 2. Chọn Mô Hình
✅ **Nên:**
- Thử nhiều kiến trúc khác nhau
- Bắt đầu với mô hình nhỏ → tăng dần
- Sử dụng pretrained models
- So sánh accuracy vs tốc độ

❌ **Không nên:**
- Chỉ dùng 1 mô hình rồi kết luận
- Dùng mô hình quá phức tạp với dữ liệu nhỏ
- Train from scratch khi có pretrained

### 3. Huấn Luyện
✅ **Nên:**
- Monitor cả train và val metrics
- Sử dụng early stopping
- Learning rate scheduling
- Save checkpoints thường xuyên

❌ **Không nên:**
- Train quá nhiều epochs
- Learning rate quá cao hoặc quá thấp
- Bỏ qua validation metrics

### 4. Đánh Giá
✅ **Nên:**
- Xem xét nhiều metrics (Accuracy, Precision, Recall, F1)
- Phân tích confusion matrix
- Kiểm tra predictions trên ảnh thực tế
- Test trên dữ liệu ngoài dataset

❌ **Không nên:**
- Chỉ nhìn Accuracy
- Bỏ qua False Positives/Negatives
- Overfit trên test set

---

## 🔧 TROUBLESHOOTING

### Vấn Đề 1: Overfitting
**Triệu chứng:**
- Train accuracy cao, Val accuracy thấp
- Gap lớn giữa train và val loss

**Giải pháp:**
1. Tăng mạnh augmentation
2. Thêm Dropout (0.3 → 0.5)
3. Giảm model complexity
4. Early stopping sớm hơn
5. Thêm regularization (L2, weight decay)

### Vấn Đề 2: Underfitting
**Triệu chứng:**
- Cả train và val accuracy đều thấp
- Loss không giảm

**Giải pháp:**
1. Tăng model complexity
2. Giảm regularization
3. Tăng số epochs
4. Tăng learning rate
5. Kiểm tra data quality

### Vấn Đề 3: Training Không Ổn Định
**Triệu chứng:**
- Loss fluctuate mạnh
- Accuracy tăng giảm liên tục

**Giải pháp:**
1. Giảm learning rate
2. Tăng batch size
3. Gradient clipping
4. Kiểm tra data preprocessing

---

## 📚 TÀI LIỆU THAM KHẢO

### Papers
1. **ResNet:** He et al. (2016) - "Deep Residual Learning for Image Recognition"
2. **EfficientNet:** Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling"
3. **VGG:** Simonyan & Zisserman (2014) - "Very Deep Convolutional Networks"

### Online Resources
- PyTorch Documentation: https://pytorch.org/docs/
- Papers with Code: https://paperswithcode.com/
- Dive into Deep Learning: https://d2l.ai/

---

## 📞 HỖ TRỢ

Nếu có thắc mắc, vui lòng:
1. Đọc kỹ phần giải thích
2. Tham khảo tài liệu gốc
3. Google error message
4. Hỏi trên Stack Overflow
5. Mở issue trên GitHub

---

**Chúc bạn học tập và nghiên cứu thành công! 🎉**
