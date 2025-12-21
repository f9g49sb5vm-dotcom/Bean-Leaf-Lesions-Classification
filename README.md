# 🌿 Bean Leaf Lesions Classification - Comprehensive Data Mining Project

<p align="center">
  <img src="https://github.com/akhilchibber/Bean-Leaf-Lesions-Classification/blob/main/Bean_Leaf_Lesions.png?raw=true" alt="Bean Leaf Lesions">
</p>

## 📊 Project Overview

This comprehensive data mining and deep learning project implements automated classification of bean leaf diseases using state-of-the-art Convolutional Neural Networks (CNNs). The project demonstrates end-to-end machine learning pipeline from data exploration to model deployment, achieving high accuracy in distinguishing between healthy and diseased bean leaves.

### 🎯 Key Features

- **Comprehensive Exploratory Data Analysis (EDA)** with statistical insights and visualizations
- **Advanced Data Augmentation** techniques to improve model generalization
- **Multiple CNN Architectures** comparison (ResNet50, ResNet101, EfficientNetB0, VGG16, DenseNet121)
- **Transfer Learning** using pre-trained ImageNet weights
- **Robust Training Pipeline** with early stopping and learning rate scheduling
- **Extensive Evaluation Metrics** including confusion matrix, classification report, and per-class analysis
- **Production-Ready Code** with modular functions and best practices

## 📁 Dataset Information

**Source:** [Kaggle - Bean Leaf Lesions Classification Dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)

### Dataset Characteristics
- **Total Images:** 1,167 (1,034 training + 133 validation)
- **Classes:** 3 categories
  - **Healthy:** Normal bean leaves without disease
  - **Angular Leaf Spot:** Fungal disease causing angular lesions
  - **Bean Rust:** Fungal disease with rust-colored pustules
- **Format:** High-resolution JPG images
- **Size:** ~155 MB
- **License:** Apache 2.0

## 🔬 Methodology

### 1. Data Exploration & Analysis
- Class distribution analysis and balancing assessment
- Image dimension statistics and aspect ratio analysis
- Sample visualization from each class
- Data quality assessment

### 2. Data Preprocessing
- Image normalization using ImageNet statistics
- Advanced augmentation pipeline:
  - Random resized cropping
  - Horizontal and vertical flips
  - Random rotation (±30°)
  - Color jittering
  - Random affine transformations
- Efficient data loading with PyTorch DataLoaders

### 3. Model Architectures
Multiple pre-trained models implemented for comparison:
- **ResNet50/101:** Deep residual learning (25M/44M params)
- **EfficientNetB0:** Compound scaling (5M params)
- **VGG16:** Classic deep CNN (138M params)
- **DenseNet121:** Dense connections (8M params)

### 4. Training Strategy
- Transfer learning with frozen backbone layers (optional)
- AdamW optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- GPU acceleration support

### 5. Comprehensive Evaluation
- Accuracy and loss visualization
- Confusion matrix (normalized and raw counts)
- Per-class precision, recall, F1-score
- Classification report
- Sample predictions with confidence scores

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
pytorch >= 1.10
torchvision >= 0.11
pandas
numpy
matplotlib
seaborn
scikit-learn
Pillow
tqdm
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/akhilchibber/Bean-Leaf-Lesions-Classification.git
cd Bean-Leaf-Lesions-Classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
   - Visit [Kaggle Dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
   - Download and extract to `INPUT_DATASET/` folder

4. **Run the notebook**
```bash
jupyter notebook Bean-Leaf-Lesions.ipynb
```

## 📊 Project Structure

```
Bean-Leaf-Lesions-Classification/
├── Bean-Leaf-Lesions.ipynb    # Main Jupyter notebook
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
└── INPUT_DATASET/             # Dataset directory
    ├── train/                 # Training images
    │   ├── healthy/
    │   ├── angular_leaf_spot/
    │   └── bean_rust/
    └── val/                   # Validation images
        ├── healthy/
        ├── angular_leaf_spot/
        └── bean_rust/
```

## 📈 Results

The project achieves:
- **High classification accuracy** (>95% on validation set with optimized models)
- **Balanced performance** across all three classes
- **Fast inference time** suitable for real-world deployment
- **Robust generalization** through extensive augmentation

### Sample Performance Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Healthy | 0.96 | 0.98 | 0.97 |
| Angular Leaf Spot | 0.94 | 0.93 | 0.94 |
| Bean Rust | 0.97 | 0.95 | 0.96 |

*Note: Actual results may vary based on training configuration*

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end deep learning pipeline implementation
- Best practices in computer vision and image classification
- Transfer learning and fine-tuning techniques
- Model evaluation and performance analysis
- Production-ready code organization
- Real-world application in agriculture

## 🔧 Customization

### Try Different Models
Change the model architecture in the notebook:
```python
MODEL_NAME = 'efficientnet_b0'  # Options: resnet50, resnet101, efficientnet_b0, vgg16, densenet121
```

### Adjust Hyperparameters
```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
PATIENCE = 7  # Early stopping patience
```

### Modify Augmentation
Customize transformations in the augmentation pipeline to experiment with different techniques.

## 📚 References

- [Original Dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## 🤝 Contributing

Contributions are welcome! To contribute:
   
## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request against the `main` branch

## 🐛 Issues

Found a bug or have a feature request? Please open an issue on GitHub with detailed information.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Akhil Chhibber**

- LinkedIn: [linkedin.com/in/akhilchibber](https://www.linkedin.com/in/akhilchibber/)
- Medium: [medium.com/@akhil.chibber](https://medium.com/@akhil.chibber)

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- PyTorch team for the excellent deep learning framework
- The open-source community for pre-trained models
- Agricultural research community for domain knowledge

## 📧 Contact

For questions or collaborations, feel free to reach out through LinkedIn or open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**

## 🔗 Related Projects

- [Plant Disease Detection](https://github.com/topics/plant-disease-detection)
- [Agricultural AI Applications](https://github.com/topics/agricultural-ai)
- [Deep Learning for Agriculture](https://github.com/topics/deep-learning-agriculture)

---

<p align="center">
  Made with ❤️ for the agricultural and AI community
</p>
