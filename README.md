# 🌱 Agri AI — Smart Farming with Artificial Intelligence

Agri AI is an **AI-powered smart agriculture platform** that helps farmers make data-driven decisions using machine learning.
The system provides intelligent tools for **crop recommendation, fertilizer guidance, pesticide detection, plant disease recognition, and crop yield prediction**.

The platform integrates **machine learning, deep learning, and federated learning** to deliver accurate agricultural insights while maintaining data privacy.

---

# 🚀 Features

### 🌾 Crop Recommendation

Recommends the best crop based on soil nutrients and environmental conditions using machine learning models.

### 🧪 Fertilizer Recommendation

Suggests optimal fertilizers based on soil nutrient composition to improve crop productivity.

### 🐛 Pesticide Recommendation

Identifies pests affecting crops and recommends appropriate pesticide treatments.

### 🍃 Plant Disease Detection

Uses a **CNN-based image classification model** to detect plant diseases from leaf images.

### 📈 Yield Prediction

Predicts expected crop yield using **federated learning models**, allowing collaborative model training while preserving farmer data privacy.

---

# 🧠 AI Technologies Used

| Technology          | Purpose                                  |
| ------------------- | ---------------------------------------- |
| Machine Learning    | Crop & fertilizer recommendation         |
| Deep Learning (CNN) | Plant disease detection                  |
| Federated Learning  | Privacy-preserving yield prediction      |
| Data Preprocessing  | Feature scaling and encoding             |
| Model Serialization | Model deployment with pickle and PyTorch |

---

# 🏗️ System Architecture

User Input → Flask Web Application → ML/DL Models → Prediction Engine → Result Display

The platform integrates multiple AI models through a **Flask backend** and an intuitive **web interface**.

---

# 🖥️ User Interface

The web interface provides a simple dashboard where farmers can access multiple AI services.

### Available Modules

* Crop Recommendation
* Fertilizer Recommendation
* Pesticide Recommendation
* Plant Disease Detection
* Yield Prediction

---

# 📂 Project Structure

```
Agri-AI
│
├── Data
│   ├── crop_yield.csv
│   └── Fertilizer Prediction.csv
│
├── federated
│   ├── client.py
│   ├── server.py
│   ├── federatedmodel.py
│   ├── train_federated.py
│
├── static
│   ├── css
│   ├── images
│   └── user_uploaded
│
├── templates
│   ├── index.html
│   ├── CropRecommendation.html
│   ├── FertilizerRecommendation.html
│   ├── PesticideRecommendation.html
│   ├── PlantDisease.html
│   └── YieldPrediction.html
│
├── app.py
├── cnn_model.py
├── Fertilizer.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation Guide

### 1️⃣ Clone the repository

```
git clone https://github.com/Rishinaiyappaag/Agri-AI.git
cd Agri-AI
```

### 2️⃣ Create a virtual environment

```
python -m venv env
```

Activate environment

Windows:

```
env\Scripts\activate
```

Mac/Linux:

```
source env/bin/activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the application

```
python app.py
```

---

# 🌍 Application Workflow

1. User opens the Agri AI web application.
2. User selects a service (crop, fertilizer, pesticide, disease, yield).
3. Input data is provided through the interface.
4. Machine learning models process the data.
5. The system returns predictions and recommendations.

---

# 🔬 Federated Learning Integration

The yield prediction model is trained using **federated learning**, enabling multiple data sources to collaboratively train models without sharing raw data.

Benefits include:

* Data privacy preservation
* Improved model generalization
* Scalable distributed training

---

# 📊 Models Used

| Model                    | Application             |
| ------------------------ | ----------------------- |
| Random Forest            | Crop recommendation     |
| Stacked ML Model         | Fertilizer prediction   |
| CNN                      | Plant disease detection |
| Federated Neural Network | Yield prediction        |

---

# 🌱 Impact

Agri AI aims to support **smart farming and sustainable agriculture** by:

* Increasing crop productivity
* Reducing fertilizer misuse
* Detecting plant diseases early
* Providing data-driven farming decisions

---

# 👨‍💻 Developed By

**Rishin Aiyappa A G**

MCA — Artificial Intelligence & Machine Learning
Jain Deemed-to-be University

---

# 📬 Connect

LinkedIn:
https://linkedin.com/

GitHub:
https://github.com/Rishinaiyappaag

---

# ⭐ If you found this project useful

Give this repository a ⭐ on GitHub and share it with others interested in AI-powered agriculture!

---

# 📜 License

This project is developed for **educational and research purposes**.
