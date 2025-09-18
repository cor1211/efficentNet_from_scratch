## 📌 Introduction
This project was built for **learning and research purposes**, focusing on the **EfficientNet architecture** in **Computer Vision**.  
Specifically, the model is trained **completely from scratch** to solve the task of **vehicle classification** (bus, family sedan, fire engine, etc.).  

---

## 📂 Dataset
- The project uses the **Vehicle Dataset from Kaggle**.  
- The dataset contains various types of vehicles for image classification tasks.   

---

## 🧠 Model Architecture
- Implements **all EfficientNet versions (B0 → B7)**.  
- The model is implemented **entirely from scratch**, **without any pretrained weights**.  
- Focuses on correctly reproducing the **EfficientNet design philosophy**:  
  - **Compound scaling**: scaling depth, width, and resolution simultaneously.  
  - **MBConv blocks**: using Depthwise Separable Convolution combined with Squeeze-and-Excitation.  

---

## 📊 Results
| Model | Accuracy |
|-------|----------|
| B0    |    81.5  | 

---

## 📎 References
- Original paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)  
- Kaggle Dataset: [Vehicle Dataset](https://www.kaggle.com/datasets/marquis03/vehicle-classification?select=train)  

---

## ⚙️ Installation
```
bash
conda env create -f environment.yml # Need to install anaconda/miniconda before
conda activate ai_env
git clone https://github.com/cor1211/efficentNet_from_scratch.git
```

---

## 🚀 Using
```
bash
python test.py --image_path "your_image_path"
```
✨ This repository is for **learning & research purposes**, designed for anyone who wants to understand how to build EfficientNet from scratch.
