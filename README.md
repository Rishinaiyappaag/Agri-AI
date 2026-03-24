# 🌱 Agri AI — Smart Farming Powered by AI

**Agri AI** is an AI-powered smart agriculture platform that empowers farmers with data-driven insights using machine learning and deep learning. The system delivers intelligent tools for crop recommendation, fertilizer guidance, pesticide identification, plant disease detection, and crop yield prediction — all through an intuitive web interface.

---

## 🚀 Features

| Module | Description |
|---|---|
| 🌾 **Crop Recommendation** | Recommends the best crop based on soil nutrients and environmental conditions |
| 🧪 **Fertilizer Recommendation** | Suggests optimal fertilizers based on soil nutrient composition |
| 🐛 **Pesticide Recommendation** | Identifies pests and recommends appropriate pesticide treatments |
| 🍃 **Plant Disease Detection** | Detects plant diseases from leaf images using CNN-based image classification |
| 📈 **Yield Prediction** | Predicts expected crop yield using federated learning for privacy-preserving collaborative training |

---

## 🧠 AI Technologies

| Technology | Purpose |
|---|---|
| Machine Learning | Crop & fertilizer recommendation |
| Deep Learning (CNN) | Plant disease detection & pesticide recommendation |
| Federated Learning | Privacy-preserving yield prediction |
| Data Preprocessing | Feature scaling and encoding |
| Model Serialization | Model deployment with Pickle and PyTorch |

---

## 📊 Models

| Model | Application |
|---|---|
| Voting Ensemble | Crop recommendation |
| Stacked Ensemble ML Model | Fertilizer prediction |
| CNN — Sequential (multi-class image classification) | Pesticide recommendation |
| CNN — DenseNet121 | Plant disease detection |
| XGBoost + Federated Learning | Yield prediction |

---

## 🏗️ System Architecture

```
User Input → Flask Web Application → ML / DL Models → Prediction Engine → Result Display
```

Multiple AI models are integrated through a **Flask backend** and served via an intuitive web interface.

---

## 📂 Project Structure

```
Agri-AI/
│
├── Data/
│   ├── crop_yield.csv
│   └── Fertilizer Prediction.csv
│
├── federated/
│   ├── client.py
│   ├── server.py
│   ├── federatedmodel.py
│   └── train_federated.py
│
├── static/
│   ├── css/
│   ├── images/
│   └── user_uploaded/
│
├── templates/
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

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rishinaiyappaag/Agri-AI.git
cd Agri-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv env
```

**Windows:**
```bash
env\Scripts\activate
```

**Mac / Linux:**
```bash
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

---

## 🌍 Application Workflow

1. Open the Agri AI web application in your browser.
2. Select a service — crop, fertilizer, pesticide, disease detection, or yield prediction.
3. Provide the required input data through the form interface.
4. The backend ML/DL models process your input.
5. Predictions and recommendations are displayed on screen.

---

## 🔬 Federated Learning — Yield Prediction

The yield prediction module is trained using **federated learning**, enabling multiple distributed data sources to collaboratively train a shared model without exposing raw data.

**Key benefits:**
- **Data privacy** — raw farm data never leaves the source device
- **Model generalization** — learns from diverse, distributed datasets
- **Scalability** — supports distributed training across multiple clients

---

## 🌱 Impact

Agri AI aims to support sustainable, smart farming by:

- Increasing crop productivity through data-driven recommendations
- Reducing fertilizer and pesticide misuse
- Enabling early detection of plant diseases
- Making AI-powered agricultural insights accessible to farmers

---

## 👨‍💻 Developer

**Rishin Aiyappa A G**  
MCA — Artificial Intelligence & Machine Learning  
Jain Deemed-to-be University

📧 [GitHub](https://github.com/Rishinaiyappaag) · [LinkedIn](https://linkedin.com/)

---

## 📜 License

This project is developed for **educational and research purposes**.

---

> ⭐ If you found this project useful, give the repository a star on GitHub and share it with others interested in AI-powered agriculture!
