# CNN for Numeric Dataset Classification

## 📌 Overview
This repository contains the implementation of a **Convolutional Neural Network (CNN)** for classifying data from the **Iris Numeric Dataset**. Unlike traditional CNN applications on image data, this project demonstrates the use of **1D CNNs for numeric feature classification**.

## 🔹 Dataset
The dataset used is the **Iris Numeric Dataset**, which consists of numerical representations of iris flower features and their respective species labels.

### **Dataset Source**
[Download from Kaggle](https://www.kaggle.com/datasets/niranjandasmm/irisnumericdatasetcsv)

### **Dataset Features**
- **Features (X):** Numeric measurements of iris flower characteristics.
- **Labels (y):** Species classification (Setosa, Versicolor, Virginica).

## 🚀 Project Structure
```
📂 CNN_NumericDataset
│── CNN_NumericDataset.ipynb  # Main notebook with CNN implementation
│── irisnumericdataset.csv    # Dataset (Downloaded from Kaggle)
│── README.md                 # Project Documentation
│── requirements.txt          # Python dependencies
```

## 🔹 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/CNN_NumericDataset.git
cd CNN_NumericDataset
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up Kaggle API (if downloading dataset manually)**
1. Download `kaggle.json` from [Kaggle API settings](https://www.kaggle.com/account).
2. Upload it to the Colab environment:
   ```python
   import os
   os.makedirs("/root/.kaggle", exist_ok=True)
   !mv kaggle.json /root/.kaggle/
   !chmod 600 /root/.kaggle/kaggle.json
   ```
3. Download the dataset:
   ```bash
   !kaggle datasets download -d niranjandasmm/irisnumericdatasetcsv
   !unzip irisnumericdatasetcsv.zip
   ```

## 🔹 Model Architecture
The CNN model consists of:
- **Conv1D Layer**: Extracts patterns from numeric sequences.
- **Flatten Layer**: Converts features into a single vector.
- **Dense Layers**: Fully connected layers for classification.
- **Dropout Layer**: Reduces overfitting.
- **Output Layer**: Uses `softmax` for multi-class classification.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

## 🔹 Training & Evaluation
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 8
- **Epochs**: 50

### **Train the Model**
```python
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
```

### **Evaluate Performance**
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)
```

## 🔹 Results
✅ CNN effectively classifies iris species using numeric features.  
✅ Demonstrates how CNNs can be applied beyond image processing.  
✅ Achieved high accuracy (~90%+).

## 📬 Contact
For questions or collaboration, reach out at **mathivarunir@gmail.com**.

