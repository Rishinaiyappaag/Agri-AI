# 🌱 AI Driven Precision Agriculture System

**AI Driven Precision Agriculture System** is an AI-powered smart agriculture platform that empowers farmers with data-driven insights using machine learning and deep learning. The system delivers intelligent tools for crop recommendation, fertilizer guidance, pesticide identification, plant disease detection, and crop yield prediction — all through an intuitive web interface with real-time camera support.

---

## 🚀 Features

| Module | Description |
|---|---|
| 🌾 **Crop Recommendation** | Recommends the best crop based on soil nutrients (NPK), temperature, humidity, pH, and rainfall with detailed reasoning showing how input values match the crop's ideal growing range |
| 🧪 **Fertilizer Recommendation** | Suggests optimal fertilizers using ML model with Ollama AI fallback, providing application rates, frequency, and precautions |
| 🐛 **Pesticide Recommendation** | Identifies pests from uploaded or camera-captured images using MobileNetV2 transfer learning and recommends appropriate pesticide treatments with dosage |
| 🍃 **Plant Disease Detection** | Detects plant diseases from leaf images using DenseNet121 with multi-layer image validation to reject non-leaf inputs (diagrams, screenshots, illustrations) |
| 📈 **Yield Prediction** | Predicts expected crop yield using XGBoost trained with federated learning for privacy-preserving collaborative training |
| 🔐 **User Authentication** | Secure login and registration system with bcrypt password hashing and session management, stored in MongoDB |
| 📷 **Real-Time Camera Support** | Both Plant Disease and Pesticide modules support live camera capture with rear-camera preference for mobile devices |

---

## 🧠 AI Technologies

| Technology | Purpose |
|---|---|
| Stacking Ensemble (XGBoost + RF + ExtraTrees + KNN) | Crop recommendation with StandardScaler pipeline |
| Stacked Ensemble ML Model | Fertilizer prediction with Ollama AI fallback |
| MobileNetV2 Transfer Learning | Pesticide recommendation (128×128, two-phase training) |
| DenseNet121 Transfer Learning | Plant disease detection (224×224) |
| XGBoost + Federated Learning | Privacy-preserving yield prediction |
| OpenCV Image Validation | Multi-layer validation to reject non-relevant images |
| Ollama (Mistral) | AI-powered fertilizer recommendation fallback |
| MongoDB | User authentication, prediction history, and session storage |

---

## 📊 Models

| Model | Architecture | Application | Key Details |
|---|---|---|---|
| Stacking Ensemble | XGBoost + Random Forest + Extra Trees + KNN → Logistic Regression meta-learner | Crop recommendation | StandardScaler pipeline, stratified 5-fold CV, sanity tests |
| Stacked Ensemble | ML model with label encoders | Fertilizer prediction | 8-feature input, Ollama AI fallback |
| MobileNetV2 | Transfer learning, 128×128 input, two-phase training | Pesticide recommendation | Phase 1: frozen base + head training, Phase 2: fine-tune top 30 layers, early stopping |
| DenseNet121 | Transfer learning, 224×224 input | Plant disease detection | Multi-layer image validation (texture, edges, color, lines, entropy) |
| XGBoost | Federated learning, 18 engineered features | Yield prediction | 98% R² accuracy, feature scaling |

---

## 🛡️ Image Validation System

Both Plant Disease and Pesticide modules include a multi-layer image validation system that rejects non-relevant inputs before the model processes them:

| Layer | Check | What It Catches |
|---|---|---|
| 1 | White background detection (>35%) | Screenshots, diagrams on white backgrounds |
| 2 | Texture analysis (Laplacian variance) | Flat illustrations, text-heavy documents |
| 3 | Straight line detection (Hough transform) | Flowcharts, ER diagrams, architecture diagrams |
| 4 | Color histogram spikiness | Images with large flat color blocks |
| 5 | Unique color count | Low-complexity illustrations |
| 6 | Saturation analysis | Grayscale or over-saturated digital art |
| 7 | Plant color presence (HSV) | Images without green/yellow/brown plant colors (Plant Disease only) |
| 8 | Model confidence threshold | Real photos of non-relevant subjects |
| 9 | Top-2 gap check | Ambiguous predictions where model can't decide |
| 10 | Entropy check | Spread-out probability distributions |

Images are rejected only when **3+ checks fail** (strike system), preventing false rejections on valid but unusual inputs like diseased leaves with spots or serrated edges.

---

## 🏗️ System Architecture

```
User Input (Form / Image Upload / Camera Capture)
         ↓
  Flask Web Application (app.py)
         ↓
  ┌────────────────────────────────────────────────────────────┐
  │              AI Prediction Engine                          │
  │                                                            │
  │  Crop: Stacking Ensemble (XGBoost+RF+ET+KNN)               │
  │  Fertilizer: ML Model → Ollama Fallback                    │
  │  Pesticide: Image Validation → MobileNetV2                 │
  │  Disease: Image Validation → DenseNet121                   │
  │  Yield: Feature Engineering → XGBoost+Federated Learning   │
  └────────────────────────────────────────────────────────────┘
         ↓                    ↕
  Result Display         MongoDB
  (with reasoning)     (Users, Predictions,
                        Recommendations)
```

---

## 📂 Project Structure

```
Agri-AI/
│
├── Data/
│   ├── crop_recommendation.csv
│   ├── crop_yield.csv
│   ├── Fertilizer Prediction.csv
│   └── train/ & test/              # Pest image dataset
│
├── federated/
│   ├── client.py
│   ├── server.py
│   ├── federatedmodel.py
│   └── train_federated.py
│
├── static/
│   ├── css/
│   ├── img/
│   │   ├── crop/                   # Crop result images
│   │   └── pesticide/              # Pest-specific product images
│   │       ├── aphids/
│   │       ├── armyworm/
│   │       ├── beetle/
│   │       ├── bollworm/
│   │       ├── earthworm/
│   │       ├── grasshopper/
│   │       ├── mites/
│   │       ├── mosquito/
│   │       ├── sawfly/
│   │       └── stem borer/
│   └── user_uploaded/              # User uploaded images
│
├── templates/
│   ├── layout.html
│   ├── index.html
│   ├── login.html
│   ├── CropRecommendation.html     # Input validation with dataset ranges
│   ├── crop-result.html            # Shows reasoning table (Your Values vs Ideal Range)
│   ├── FertilizerRecommendation.html
│   ├── Fertilizer-Result.html
│   ├── PesticideRecommendation.html  # Upload + Camera tabs
│   ├── PestNotRecognized.html      # Shown when pest image is rejected
│   ├── PlantDisease.html           # Upload + Camera tabs
│   ├── PlantDiseaseResult.html     # Shows valid/invalid with reasoning
│   ├── YieldPrediction.html
│   ├── aphids.html                 # Pest-specific result pages
│   ├── armyworm.html
│   ├── beetle.html
│   ├── bollworm.html
│   ├── earthworm.html
│   ├── grasshopper.html
│   ├── mites.html
│   ├── mosquito.html
│   ├── sawfly.html
│   ├── stem borer.html
│   ├── unaptfile.html
│   └── error.html
│
├── utils/
│   └── fertilizer.py
│
├── app.py                          # Main Flask application
├── auth.py                         # User authentication (bcrypt)
├── database.py                     # MongoDB schema & initialization
│
├── plant_disease_predictor.py      # DenseNet121 + image validation
├── pest_predictor.py               # MobileNetV2 + image validation
│
├── train_plant_disease_densenet.py # DenseNet121 training script
├── train_crop_recommendation.py    # Stacking ensemble training script
├── pesticide_recomendation.py      # MobileNetV2 pest training script
│
├── Crop_Recommendation.pkl         # Trained crop model (Pipeline: Scaler + Stacking)
├── crop_stats.json                 # Crop feature statistics for result reasoning
├── Fertilizer_Stack_Model.pkl      # Trained fertilizer model
├── soil_encoder.pkl                # Soil type label encoder
├── crop_encoder.pkl                # Crop type label encoder
├── fertilizer_encoder.pkl          # Fertilizer label encoder
├── Trained_model.h5                # Trained pest CNN (MobileNetV2)
├── pest_class_names.json           # Pest class names and image size config
├── plant_disease_densenet.pth      # Trained plant disease model (DenseNet121)
├── federated_yield_model.pth       # Trained yield model (XGBoost federated)
│
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

**Key dependencies:**
```
flask
flask-pymongo
tensorflow==2.15.0
torch
torchvision
xgboost
opencv-python
numpy==1.26.4
ml-dtypes==0.2.0
scikit-learn
pandas
pillow
bcrypt
python-dotenv
```

> **Important:** NumPy, TensorFlow, and ml-dtypes versions must be compatible. If you encounter `_ARRAY_API not found` errors, run:
> ```bash
> pip install numpy==1.26.4 --force-reinstall
> pip install tensorflow==2.15.0 --force-reinstall
> pip install ml-dtypes==0.2.0 --no-deps
> ```

### 4. Configure MongoDB

Make sure MongoDB is running locally or provide a connection URI in `.env`:

```
MONGO_URI=mongodb://localhost:27017/agri_ai_db
SECRET_KEY=your-secret-key
```

### 5. Train the models (if not already trained)

```bash
# Crop recommendation (Stacking Ensemble)
python train_crop_recommendation.py

# Pest classification (MobileNetV2)
python pesticide_recomendation.py

# Plant disease detection (DenseNet121)
python train_plant_disease_densenet.py
```

### 6. Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5000/login` in your browser.

---

## 🌍 Application Workflow

1. **Register / Login** — Credentials are securely stored in MongoDB with bcrypt password hashing.
2. **Select a service** — Crop, Fertilizer, Pesticide, Plant Disease, or Yield Prediction.
3. **Provide input** — Fill forms with validated ranges, upload images, or use the live camera.
4. **AI processes your input** — Multi-layer validation filters invalid images, then ML/DL models generate predictions.
5. **View results with reasoning** — Crop results show a comparison table of your values vs the crop's ideal range. Disease/pest results show confidence scores and validation status.
6. **Download reports** — Crop recommendation results include a downloadable text report.

---

## 🍃 MongoDB — Database Schema (ERD)

| Collection | Purpose | Key Fields |
|---|---|---|
| `users` | User accounts | name, email, password (bcrypt), role, created_at |
| `crop_recommendations` | Crop prediction history | user_id, N, P, K, pH, rainfall, temperature, humidity, recommended_crop |
| `fertilizer_recommendations` | Fertilizer prediction history | user_id, N, P, K, crop_type, recommended_fertilizer |
| `pesticide_recommendations` | Pest prediction history | user_id, image_path, predicted_pest, confidence_score |
| `disease_predictions` | Disease prediction history | user_id, image_path, predicted_disease, confidence_score |
| `yield_predictions` | Yield prediction history | user_id, area, rainfall, fertilizer, pesticide, predicted_yield |

---

## 🔬 Federated Learning — Yield Prediction

The yield prediction module is trained using **federated learning**, enabling multiple distributed data sources to collaboratively train a shared XGBoost model without exposing raw data.

**Key benefits:**
- **Data privacy** — Raw farm data never leaves the source device
- **Model generalization** — Learns from diverse, distributed datasets
- **High accuracy** — 98% R² with 18 engineered features including log transforms, interaction terms, and squared features

---

## 📷 Camera Support

Both Plant Disease and Pesticide modules support real-time camera capture:

- Uses `navigator.mediaDevices.getUserMedia()` with rear-camera preference for mobile
- Camera images are sanitized through PIL (forced RGB, EXIF stripping) before prediction
- Captured images go through the same validation and prediction pipeline as uploaded images

---

## 🌱 Impact

AI Driven Precision Agriculture System aims to support sustainable, smart farming by:

- Increasing crop productivity through data-driven recommendations with transparent reasoning
- Reducing fertilizer and pesticide misuse with targeted AI guidance
- Enabling early detection of plant diseases through image analysis
- Preventing false predictions with multi-layer image validation
- Making AI-powered agricultural insights accessible to farmers through camera support

---

## 👨‍💻 Developer

**Rishin Aiyappa A G**
MCA — Artificial Intelligence & Machine Learning
Jain Deemed-to-be University

---

## 📜 License

This project is developed for **educational and research purposes**.
