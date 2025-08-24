# 🌱 Crop Disease Classifier

**AI-Powered Plant Disease Detection Web Application** <br>
A web application leveraging TensorFlow, Keras, and Streamlit to assist farmers and researchers in identifying plant diseases from leaf images. Users can upload crop images to receive accurate disease predictions with confidence scores, all through a clean and intuitive interface.

<img width="1875" height="952" alt="Screenshot (190)" src="https://github.com/user-attachments/assets/fa321405-9748-4937-a369-f65e81edb85f" />

<p align="center">
  <a href="https://cropdiseaseclassifier.streamlit.app/">
    <img src="https://img.shields.io/badge/LIVE-VISIT%20NOW-blue?style=for-the-badge&logo=streamlit" alt="Live">
  </a>
</p>

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
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/) [![Keras](https://img.shields.io/badge/Keras-3.x-D00000.svg)](https://keras.io/) [![Pillow](https://img.shields.io/badge/Pillow-9.x-3670A0.svg)](https://python-pillow.org/)  [![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b.svg)](https://streamlit.io/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
- **Python 3.9+** - Core programming language
- [TensorFlow / Keras](https://www.tensorflow.org/) – Deep learning model  
- [NumPy](https://numpy.org/) – Data processing  
- [Pillow](https://pillow.readthedocs.io/) – Image handling  
- [Streamlit](https://streamlit.io/) – Web application framework
- [FastAPI](https://fastapi.tiangolo.com/) – Backend implementation for handling model inference and API requests.

---

## 📂 Project Structure

```
Crop_Disease_Classifier/
├── corn/
│   ├── checkpoints/                          # Saved models for each epoch
│   ├── corn_disease/                         # Corn disease dataset
│   ├── saved_models/
│   │   └── corn_disease_model.h5             # Trained CNN model
│   └── corn_disease_classification.ipynb     # Model training notebook
│
├── potato/
│   ├── checkpoints/                          # Saved models for each epoch
│   ├── potato_disease/                       # Potato disease dataset
│   ├── saved_models/
│   │   └── potato_disease_model.h5           # Trained CNN model
│   └── potato_disease_classification.ipynb   # Model training notebook
│
├── rice/
│   ├── checkpoints/                          # Saved models for each epoch
│   ├── rice_disease/                         # Rice disease dataset
│   ├── saved_models/
│   │   └── rice_disease_model.h5             # Trained CNN model
│   └── rice_disease_classification.ipynb     # Model training notebook
│
├── wheat/
│   ├── checkpoints/                          # Saved models for each epoch
│   ├── wheat_disease/                        # Wheat disease dataset
│   ├── saved_models/
│   │   └── wheat_disease_model.h5            # Trained CNN model
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
git clone https://github.com/niloysannyal/Crop_Disease_Classifier.git
cd Crop_Disease_Classifier
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
- Batch size: 32
- Output: Predicted disease class & confidence score
- Training data: Public crop disease dataset (PlantVillage or similar)

---

## 🎯 Usage
1. Select Crop type (Corn, Potato, Rice, Wheat)
2. Upload a crop leaf image (.jpg, .jpeg, .png, .webp)
3. Click 🔍 Predict Disease
4. View prediction result with confidence score
5. Click 🔄 Try Again to test another image

https://github.com/user-attachments/assets/bbfb7169-8434-42eb-bb8d-799f46b1107c

---

## 📸 Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/1b8e19a6-5327-49b8-bb8e-e2c3b21bbcb2" alt="Image 1" width="22%">
  <img src="https://github.com/user-attachments/assets/ea576a17-28bb-400b-aa5f-da771da2f235" alt="Image 2" width="22%">
  <img src="https://github.com/user-attachments/assets/750e2e3f-6e85-4788-b2e7-2a33ed341f3d" alt="Image 3" width="22%">
  <img src="https://github.com/user-attachments/assets/a4f78ffb-4b12-4508-951a-3e5a45518614" alt="Image 4" width="22%">
</p>

---

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

🔗 [Portfolio](https://niloysannyal.github.io/Portfolio/) | [LinkedIn](https://www.linkedin.com/in/niloysannyal) | [GitHub](https://github.com/niloysannyal)  
