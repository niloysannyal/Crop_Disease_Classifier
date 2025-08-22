# 🌾 Crop Disease Classifier

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)  [![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b.svg)](https://streamlit.io/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

An AI-powered web application built with **TensorFlow**, **Keras**, and **Streamlit** to help farmers and researchers detect plant diseases from leaf images.  
Upload a crop image, and the app predicts the disease class with confidence, providing a simple and user-friendly interface.

---

## 🚀 Features
- 📸 Upload crop leaf images for disease detection  
- 🔍 Real-time **disease classification** with confidence score  
- 🎨 Modern and intuitive **Streamlit UI**  
- 🔄 Single-button workflow for **Predict** / **Try Again**  
- ⚡ Optimized for **fast predictions**  
- 🖥️ Cross-platform: works locally and deployable on **Streamlit Cloud**, **Render**, or **Heroku**  

---

## 🛠️ Tech Stack
- **Python 3.9+**
- [TensorFlow / Keras](https://www.tensorflow.org/) – Deep learning model  
- [NumPy](https://numpy.org/) – Data processing  
- [Pillow](https://pillow.readthedocs.io/) – Image handling  
- [Streamlit](https://streamlit.io/) – Web application framework  

---

## 📂 Project Structure

```
Crop_Disease_Classifier/
├── corn/
│   ├── checkpoints/                          # Saved models for each epoch
│   ├── corn_disease/                         # Corn dataset
│   ├── saved_models/
│   │   └── corn_disease_model.h5             # Pre-trained CNN model
│   └── corn_disease_classification.ipynb     # Model training notebook
│
├── potato/
│   ├── checkpoints/                          # Saved models for each epoch
│   ├── potato_disease/                       # Potato dataset
│   ├── saved_models/
│   │   └── potato_disease_model.h5           # Pre-trained CNN model
│   └── potato_disease_classification.ipynb   # Model training notebook
│
├── rice/
│   ├── checkpoints/                          # Saved models for each epoch
│   ├── rice_disease/                         # Rice dataset
│   ├── saved_models/
│   │   └── rice_disease_model.h5             # Pre-trained CNN model
│   └── rice_disease_classification.ipynb     # Model training notebook
│
├── wheat/
│   ├── checkpoints/                          # Saved models for each epoch
│   ├── wheat_disease/                        # Wheat dataset
│   ├── saved_models/
│   │   └── wheat_disease_model.h5            # Pre-trained CNN model
│   └── wheat_disease_classification.ipynb    # Model training notebook
│
├── main.py                                   # FastAPI backend for experiment
├── app.py                                    # Streamlit application
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
└── LICENCE                                   # MIT licence
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/CropDiseaseClassifier.git
cd CropDiseaseClassifier
```

### 2. Create a virtual environment
```
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the app
```
streamlit run app.py
```

---

## 📊 Model Details
- Architecture: Convolutional Neural Network (CNN)
- Input size: 224x224 pixels (RGB)
- Output: Predicted disease class & confidence score
- Training data: Public crop disease dataset (PlantVillage or similar)


## 🎯 Usage
1. Select Crop type (Corn, Potato, Rice, Wheat)
2. Upload a crop leaf image (.jpg, .jpeg, .png, .webp)
3. Click 🔍 Predict Disease
4. View prediction result with confidence score
5. Click 🔄 Try Again to test another image


## 📸 Screenshots
(App Screenshots will be added here)


## 🤝 Contributing  
Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](../../issues).  

---

## 📜 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

## 👨‍💻 Author  
**Niloy Sannyal**  
📍 Dhaka, Bangladesh  
📧 [niloysannyal@gmail.com](mailto:niloysannyal@gmail.com)  

🔗 [Portfolio](#) | [LinkedIn](#) | [GitHub](#)  
