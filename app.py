"""
app.py - AGRI AI with Authentication System
Complete app with login, signup, and MongoDB integration
+ Unknown image rejection for Plant Disease & Pesticide
+ Camera support for both modules
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_pymongo import PyMongo
from markupsafe import Markup
import numpy as np
import pickle
import requests
import os
import json
import pandas as pd
import warnings
from dotenv import load_dotenv
from auth import create_user, verify_login, get_user_by_email, hash_password
from database import init_db, new_fertilizer_rec, new_pesticide_rec, new_disease_prediction, new_yield_prediction, new_crop_recommendation
from functools import wraps
from datetime import datetime
from bson.objectid import ObjectId
from PIL import Image
import io

warnings.filterwarnings('ignore')
load_dotenv()

# ╔════════════════════════════════════════════════════════════════════╗
# ║                    FLASK APP INITIALIZATION                       ║
# ╚════════════════════════════════════════════════════════════════════╝

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-this')
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/agri_ai_db')

mongo = PyMongo(app)

with app.app_context():
    init_db(db_name="agri_ai_db")

UPLOAD_FOLDER = "static/user_uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ╔════════════════════════════════════════════════════════════════════╗
# ║                    OLLAMA CONFIGURATION                           ║
# ╚════════════════════════════════════════════════════════════════════╝

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
OLLAMA_ENABLED = True

# ╔════════════════════════════════════════════════════════════════════╗
# ║                    LOAD ALL MODELS                                ║
# ╚════════════════════════════════════════════════════════════════════╝

print("\n" + "="*70)
print("LOADING MODELS - AGRI AI SYSTEM")
print("="*70)

# ─── Version diagnostics ───
print(f"\n  NumPy version: {np.__version__}")
try:
    import tensorflow as tf
    print(f"  TensorFlow version: {tf.__version__}")
    TF_AVAILABLE = True
except Exception as e:
    print(f"  ⚠️  TensorFlow import FAILED: {e}")
    print(f"  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  FIX: Run these commands in your terminal:              ║")
    print(f"  ║  pip install numpy==1.24.3 --force-reinstall            ║")
    print(f"  ║  pip install tensorflow==2.15.0 --force-reinstall       ║")
    print(f"  ║  pip install ml-dtypes==0.2.0 --force-reinstall         ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    TF_AVAILABLE = False
    tf = None

print()

# ─── Crop Recommendation Model ───
try:
    with open("Crop_Recommendation.pkl", "rb") as f:
        crop_recommendation_model = pickle.load(f)
    print("✓ Crop Recommendation Model loaded")
except Exception as e:
    print(f"⚠️  Crop Recommendation Model not found: {e}")
    crop_recommendation_model = None

# ─── Fertilizer Model (8 Features) ───
try:
    with open("Fertilizer_Stack_Model.pkl", "rb") as f:
        fertilizer_model = pickle.load(f)
    print("✓ Fertilizer Model loaded (8 features)")
    FERTILIZER_ML_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Fertilizer Model not found - Will use Ollama fallback")
    fertilizer_model = None
    FERTILIZER_ML_AVAILABLE = False

# ─── Encoders ───
try:
    soil_encoder = pickle.load(open("soil_encoder.pkl", "rb"))
    crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))
    fertilizer_encoder = pickle.load(open("fertilizer_encoder.pkl", "rb"))
    print("✓ Encoders loaded")
    ENCODERS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Encoders not found - Will use Ollama fallback")
    soil_encoder = None
    crop_encoder = None
    fertilizer_encoder = None
    ENCODERS_AVAILABLE = False

# ─── YIELD MODEL (XGBoost Federated Learning) ───
try:
    with open("federated_yield_model.pth", "rb") as f:
        yield_model_package = pickle.load(f)
    yield_model     = yield_model_package["model"]
    yield_scaler    = yield_model_package["scaler"]
    feature_cols    = yield_model_package["feature_cols"]
    metrics         = yield_model_package["metrics"]
    print(f"✓ Yield Model (XGBoost Federated Learning) loaded")
    print(f"  Features: {len(feature_cols)} | Test R²: {metrics.get('test_r2', 0):.6f}")
    YIELD_MODEL_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Yield Model not found: {e}")
    yield_model  = None
    yield_scaler = None
    feature_cols = None
    metrics      = None
    YIELD_MODEL_AVAILABLE = False

# ─── Pest Model (SEPARATED loading — TF, model file, predictor) ───
pest_model = None
tf_image = None
PEST_MODEL_AVAILABLE = False

if TF_AVAILABLE:
    # Step 1: Import keras image helper
    try:
        from tensorflow.keras.preprocessing import image as tf_image
    except Exception as e:
        print(f"⚠️  keras.preprocessing.image import failed: {e}")
        tf_image = None

    # Step 2: Load the .h5 model file (try multiple case variants)
    pest_model_candidates = [
        "Trained_model.h5",
        "Trained_Model.h5",
        "trained_model.h5",
        "Trained_model.H5",
    ]
    for model_file in pest_model_candidates:
        if os.path.exists(model_file):
            try:
                pest_model = tf.keras.models.load_model(model_file)
                print(f"✓ Pest Model loaded from: {model_file}")
                break
            except Exception as e:
                print(f"⚠️  Failed to load {model_file}: {e}")

    if pest_model is None:
        print(f"⚠️  Pest Model .h5 file not found! Looked for: {pest_model_candidates}")
        print(f"   Make sure the file exists in: {os.getcwd()}")

    # Step 3: Import pest_predictor module
    if pest_model is not None:
        try:
            from pest_predictor import predict_pest, predict_pest_from_pil
            PEST_MODEL_AVAILABLE = True
            print("✓ Pest Predictor module loaded (with unknown detection + camera)")
        except Exception as e:
            print(f"⚠️  pest_predictor.py import failed: {e}")
            print(f"   Make sure pest_predictor.py exists in: {os.getcwd()}")
            PEST_MODEL_AVAILABLE = False
else:
    print("⚠️  Pest Model skipped — TensorFlow not available")

# ─── Plant Disease Model (with validation) ───
try:
    from plant_disease_predictor import load_model as load_plant_model
    from plant_disease_predictor import predict as predict_plant_disease
    from plant_disease_predictor import predict_from_pil as predict_plant_from_pil
    PLANT_MODEL_PATH = "plant_disease_densenet.pth"
    plant_model, plant_classes = load_plant_model(PLANT_MODEL_PATH)
    print("✓ Plant Disease Model loaded (with unknown detection + camera)")
except Exception as e:
    print(f"⚠️  Plant Disease Model not found: {e}")
    plant_model   = None
    plant_classes = None

# ─── Fertilizer Data ───
try:
    from utils.fertilizer import fertilizer_dict
    print("✓ Fertilizer data loaded")
except Exception as e:
    print(f"⚠️  Fertilizer data not found")
    fertilizer_dict = {}

print("="*70 + "\n")

# ╔════════════════════════════════════════════════════════════════════╗
# ║                    CROPS & SOILS LIST                             ║
# ╚════════════════════════════════════════════════════════════════════╝

crops = [
    "Rice", "Wheat", "Maize", "Barley", "Bajra", "Ragi",
    "Small millets", "Gram", "Arhar/Tur", "Urad",
    "Peas & beans (Pulses)", "Groundnut", "Soyabean",
    "Rapeseed & Mustard", "Sesamum", "Sunflower", "Castor seed",
    "Sugarcane", "Cotton(lint)", "Potato", "Sweet potato",
    "Onion", "Garlic", "Ginger", "Turmeric", "Coriander",
    "Dry chillies", "Banana", "Coconut", "Tapioca", "Areca nut", "Cashewnut"
]

soils = [
    "Black", "Red", "Loamy", "Sandy", "Clayey", "Laterite", "Alluvial", "Peaty"
]

# ╔════════════════════════════════════════════════════════════════════╗
# ║                    FERTILIZER DATABASE                            ║
# ╚════════════════════════════════════════════════════════════════════╝

FERTILIZER_DATABASE = {
    "14-35-14": {
        "name": "DAP (Diammonium Phosphate)",
        "rate": "150-200 kg/hectare",
        "frequency": "Once during basal application at planting",
        "reason": "High phosphorous (35%) and nitrogen (14%) content promotes strong flowering and fruiting.",
        "tips": ["Mix thoroughly with soil before planting", "Apply uniformly across the field", "Can be applied with or without irrigation", "Best results when applied in slightly moist soil", "Monitor crop for nutrient deficiency symptoms"],
        "precautions": "Not suitable for highly acidic soils. Avoid application in dry conditions."
    },
    "10-52-34": {
        "name": "NPK Compound (10-52-34)",
        "rate": "200-250 kg/hectare",
        "frequency": "Basal application + Split application after 4-6 weeks",
        "reason": "Balanced NPK with higher P and K. Excellent for root crops and tubers.",
        "tips": ["Apply as basal dose at planting", "Follow up with split application during growth", "Ensure soil moisture before application", "Mix with other fertilizers for better coverage", "Water field lightly after application"],
        "precautions": "Suitable for most soils. May cause chloride accumulation in coastal areas."
    },
    "20-20-20": {
        "name": "Balanced NPK (20-20-20)",
        "rate": "300-400 kg/hectare",
        "frequency": "Split application in 3 doses",
        "reason": "Equal nitrogen, phosphorous, and potassium for all crop growth stages.",
        "tips": ["Divide total dose into 3 equal parts", "Apply first dose at planting (basal)", "Apply second dose at 30 days growth", "Apply third dose at 60 days or flowering stage", "Ensure irrigation before and after each application"],
        "precautions": "Suitable for all crops and soils. May require micronutrient supplementation."
    },
    "46-0-0": {
        "name": "Urea (Nitrogen fertilizer)",
        "rate": "100-150 kg/hectare",
        "frequency": "Split application in 2-3 doses at 4-week intervals",
        "reason": "Pure nitrogen source for rapid vegetative growth and green foliage.",
        "tips": ["Always apply in split doses to reduce loss", "First dose at active growth stage (20-30 days)", "Second dose at 4-6 week intervals", "Dissolve in water for even distribution", "Apply in moist soil for better uptake"],
        "precautions": "Risk of nitrate leaching in sandy soils. Do not exceed 150 kg/ha."
    },
    "0-46-0": {
        "name": "Superphosphate (SSP)",
        "rate": "200-250 kg/hectare",
        "frequency": "Once as basal application before or at planting",
        "reason": "Pure phosphorous source for root development and flowering.",
        "tips": ["Apply as basal dose mixed with soil", "Mix evenly to ensure uniform distribution", "Suitable for acidic and neutral soils", "Contains sulfur beneficial for cruciferous crops", "Can be combined with manure for better results"],
        "precautions": "May increase soil acidity slightly. Not recommended for alkaline soils."
    }
}

# ╔════════════════════════════════════════════════════════════════════╗
# ║                    LOGIN DECORATOR                                ║
# ╚════════════════════════════════════════════════════════════════════╝

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user_id():
    return ObjectId(session['user_id'])


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    AUTHENTICATION ROUTES                          ║
# ╚════════════════════════════════════════════════════════════════════╝

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        action = request.form.get("action")

        if action == "signup":
            full_name        = request.form.get("signup_name", "").strip()
            email            = request.form.get("signup_email", "").strip()
            password         = request.form.get("signup_password", "").strip()
            confirm_password = request.form.get("signup_confirm", "").strip()

            if not full_name or not email or not password:
                return render_template("login.html", error="❌ All fields are required!", active_tab="signup")
            if len(password) < 6:
                return render_template("login.html", error="❌ Password must be at least 6 characters!", active_tab="signup")
            if password != confirm_password:
                return render_template("login.html", error="❌ Passwords do not match!", active_tab="signup")
            if "@" not in email:
                return render_template("login.html", error="❌ Invalid email address!", active_tab="signup")

            success, message, user_id = create_user(mongo, full_name, email, password)
            if success:
                return render_template("login.html", success=message, active_tab="signin")
            else:
                return render_template("login.html", error=message, active_tab="signup")

        elif action == "signin":
            email    = request.form.get("signin_email", "").strip()
            password = request.form.get("signin_password", "").strip()

            if not email or not password:
                return render_template("login.html", error="❌ Email and password are required!", active_tab="signin")

            success, message, user = verify_login(mongo, email, password)
            if success:
                session['user_id']    = str(user['_id'])
                session['user_name']  = user['name']
                session['user_email'] = user['email']
                session['user_role']  = user['role']
                return redirect(url_for('index'))
            else:
                return render_template("login.html", error=message, active_tab="signin")

    return render_template("login.html", active_tab="signin")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    PAGE ROUTES (Protected)                        ║
# ╚════════════════════════════════════════════════════════════════════╝

@app.route("/")
@app.route("/index.html")
@login_required
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
@login_required
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
@login_required
def fertilizer_page():
    try:
        soils_list = soil_encoder.classes_ if soil_encoder else soils
        crops_list = crop_encoder.classes_ if crop_encoder else crops
    except Exception:
        soils_list = soils
        crops_list = crops
    return render_template("FertilizerRecommendation.html", crops=crops_list, soils=soils_list, prediction=None, form_data=None, error=None)

@app.route("/PesticideRecommendation.html")
@login_required
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/PlantDisease.html")
@login_required
def plant_disease_page():
    return render_template("PlantDisease.html")

@app.route("/yield")
@app.route("/YieldPrediction.html")
@login_required
def yield_page():
    return render_template("YieldPrediction.html", crops=crops, prediction=None, form_data=None, error=None)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    CROP PREDICTION                                ║
# ╚════════════════════════════════════════════════════════════════════╝

# Load crop statistics for result page reasoning
CROP_STATS = {}
try:
    with open("crop_stats.json", "r") as f:
        CROP_STATS = json.load(f)
    print("✓ Crop statistics loaded for result reasoning")
except Exception:
    print("⚠️  crop_stats.json not found — run train_crop_recommendation.py first")


@app.route("/crop_prediction", methods=["POST"])
@login_required
def crop_prediction():
    nitrogen     = float(request.form["nitrogen"])
    phosphorous  = float(request.form["phosphorous"])
    potassium    = float(request.form["potassium"])
    temperature  = float(request.form["temperature"])
    humidity     = float(request.form["humidity"])
    ph           = float(request.form["ph"])
    rainfall     = float(request.form["rainfall"])

    data = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
    prediction = crop_recommendation_model.predict(data)[0]

    # Build reasoning from crop stats
    input_values = {
        "N": nitrogen, "P": phosphorous, "K": potassium,
        "temperature": temperature, "humidity": humidity,
        "ph": ph, "rainfall": rainfall
    }

    crop_info = CROP_STATS.get(prediction, {})

    try:
        doc = new_crop_recommendation(
            user_id=get_current_user_id(), nitrogen=nitrogen, phosphorus=phosphorous,
            potassium=potassium, soil_ph=ph, rainfall=rainfall,
            temperature=temperature, humidity=humidity, recommended_crop=prediction
        )
        mongo.db.crop_recommendations.insert_one(doc)
    except Exception as e:
        print(f"⚠️  Could not save crop recommendation: {e}")

    return render_template(
        "crop-result.html",
        prediction=prediction,
        pred="img/crop/" + prediction + ".jpg",
        input_values=input_values,
        crop_info=crop_info
    )


# ╔════════════════════════════════════════════════════════════════════╗
# ║    FERTILIZER RECOMMENDATION (HYBRID: ML + OLLAMA FALLBACK)       ║
# ╚════════════════════════════════════════════════════════════════════╝

def call_ollama(prompt):
    try:
        response = requests.post(OLLAMA_API_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "temperature": 0.7}, timeout=60)
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        return None
    except Exception:
        return None


def format_ml_fertilizer_recommendation(fertilizer_code):
    if fertilizer_code in FERTILIZER_DATABASE:
        info = FERTILIZER_DATABASE[fertilizer_code]
        rec = f"RECOMMENDED FERTILIZER (Primary): {info['name']} ({fertilizer_code})\n\nAPPLICATION RATE: {info['rate']}\n\nAPPLICATION FREQUENCY: {info['frequency']}\n\nWHY THIS FERTILIZER:\n{info['reason']}\n\nAPPLICATION TIPS:\n"
        for i, tip in enumerate(info['tips'], 1):
            rec += f"{i}. {tip}\n"
        rec += f"\nPRECAUTIONS: {info['precautions']}"
        return rec
    else:
        return f"RECOMMENDED FERTILIZER (Primary): {fertilizer_code}\n\nThis is a specialized fertilizer blend recommended by our ML model.\n\nAPPLICATION RATE: Please consult with a local agricultural expert for the exact rate.\n\nFREQUENCY: Typically applied as basal dose + split application during growth period."


@app.route("/fertilizer", methods=["POST"])
@login_required
def recommend_fertilizer():
    try:
        soils_list = soil_encoder.classes_ if soil_encoder else soils
        crops_list = crop_encoder.classes_ if crop_encoder else crops
    except Exception:
        soils_list = soils
        crops_list = crops

    try:
        temperature  = float(request.form.get("temperature", 25))
        humidity     = float(request.form.get("humidity", 50))
        moisture     = float(request.form.get("moisture", 50))
        soil         = request.form.get("soil", "").strip()
        cropname     = request.form.get("cropname", "").strip()
        nitrogen     = float(request.form.get("nitrogen", 50))
        phosphorous  = float(request.form.get("phosphorous", 50))
        potassium    = float(request.form.get("potassium", 50))

        if not soil or not cropname:
            return render_template("FertilizerRecommendation.html", prediction=None, crops=crops_list, soils=soils_list, form_data=request.form, error="❌ Please select both Soil and Crop!")

        recommended_fertilizer = None
        fertilizer_code = "unknown"
        method_used = ""

        if FERTILIZER_ML_AVAILABLE and ENCODERS_AVAILABLE:
            try:
                soil_encoded = soil_encoder.transform([soil])[0]
                crop_encoded = crop_encoder.transform([cropname])[0]
                input_data = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded, nitrogen, phosphorous, potassium]], dtype=np.float32)
                fertilizer_prediction = fertilizer_model.predict(input_data)[0]
                fertilizer_code = str(fertilizer_encoder.inverse_transform([int(fertilizer_prediction)])[0]).strip()
                recommended_fertilizer = format_ml_fertilizer_recommendation(fertilizer_code)

                alternatives = []
                for code, info in list(FERTILIZER_DATABASE.items())[:2]:
                    if code != fertilizer_code:
                        alt = f"\n\n---\n\nALTERNATIVE OPTION {len(alternatives)+1}: {info['name']} ({code})\n\nAPPLICATION RATE: {info['rate']}\nREASON: {info['reason']}\nAPPLICATION TIPS:\n"
                        for i, tip in enumerate(info['tips'][:3], 1):
                            alt += f"{i}. {tip}\n"
                        recommended_fertilizer += alt
                        alternatives.append(code)
                        if len(alternatives) == 2:
                            break
                method_used = "ML Model"
            except Exception as e:
                print(f"ML Model failed: {e}")
                recommended_fertilizer = None

        if recommended_fertilizer is None and OLLAMA_ENABLED:
            try:
                prompt = f"Recommend 2-3 different fertilizers for:\nCrop: {cropname}, Soil: {soil}, Temp: {temperature}°C, N:{nitrogen}, P:{phosphorous}, K:{potassium}"
                recommended_fertilizer = call_ollama(prompt)
                if recommended_fertilizer:
                    fertilizer_code = "ollama"
                    method_used = "Ollama AI"
            except Exception as e:
                print(f"Ollama failed: {e}")

        if recommended_fertilizer is None:
            return render_template("FertilizerRecommendation.html", prediction=None, crops=crops_list, soils=soils_list, form_data=request.form, error="❌ Could not generate recommendation.")

        try:
            doc = new_fertilizer_rec(user_id=get_current_user_id(), nitrogen=nitrogen, phosphorus=phosphorous, potassium=potassium, crop_type=cropname, recommended_fertilizer=fertilizer_code)
            mongo.db.fertilizer_recommendations.insert_one(doc)
        except Exception as e:
            print(f"⚠️  Could not save fertilizer recommendation: {e}")

        return render_template("Fertilizer-Result.html", prediction=recommended_fertilizer, method=method_used, crops=crops_list, soils=soils_list, form_data=request.form, error=None)

    except Exception as e:
        return render_template("FertilizerRecommendation.html", prediction=None, crops=crops_list, soils=soils_list, form_data=request.form, error=f"❌ Error: {str(e)}")


# ╔════════════════════════════════════════════════════════════════════╗
# ║       YIELD PREDICTION (XGBoost + PRODUCTION + 18 FEATURES)       ║
# ╚════════════════════════════════════════════════════════════════════╝

@app.route("/predict_yield", methods=["POST"])
@login_required
def predict_yield():
    if not YIELD_MODEL_AVAILABLE:
        return render_template("YieldPrediction.html", prediction=None, crops=crops, form_data=request.form, error="❌ Yield model not loaded!")

    try:
        crop_name  = request.form.get("Crop", "Rice")
        year       = float(request.form.get("Crop_Year", 2025))
        area       = float(request.form.get("Area", 1))
        production = float(request.form.get("Production", 50))
        rainfall   = float(request.form.get("Annual_Rainfall", 600))
        fertilizer = float(request.form.get("Fertilizer", 100))
        pesticide_val = float(request.form.get("Pesticide", 50))

        area       = max(area, 1)
        production = max(production, 1)
        if year < 1990 or year > 2030:
            year = 2025
        rainfall   = max(0, rainfall)
        fertilizer = max(0, fertilizer)
        pesticide_val = max(0, pesticide_val)

        f_area_log        = np.log1p(area)
        f_production_log  = np.log1p(production)
        f_fertilizer_log  = np.log1p(fertilizer + 1)
        f_pesticide_log   = np.log1p(pesticide_val + 1)
        f_fert_pest       = fertilizer * pesticide_val
        f_area_rain       = area * rainfall
        f_rain_fert       = rainfall * fertilizer
        f_fert_area       = fertilizer / (area + 1)
        f_pest_area       = pesticide_val / (area + 1)
        f_prod_area       = production / (area + 1)
        f_area_sq         = area ** 2
        f_rain_sq         = rainfall ** 2

        features = np.array([[year, area, production, rainfall, fertilizer, pesticide_val, f_area_log, f_production_log, f_fertilizer_log, f_pesticide_log, f_fert_pest, f_area_rain, f_rain_fert, f_fert_area, f_pest_area, f_prod_area, f_area_sq, f_rain_sq]], dtype=np.float32)

        features_scaled  = yield_scaler.transform(features)
        yield_prediction = float(yield_model.predict(features_scaled)[0])
        yield_prediction = max(0.1, round(yield_prediction, 4))

        yield_per_hectare    = yield_prediction
        yield_per_acre       = round(yield_prediction / 2.471, 4)
        estimated_production = round(yield_prediction * area, 2)

        try:
            doc = new_yield_prediction(user_id=get_current_user_id(), area=area, annual_rainfall=rainfall, fertilizer=fertilizer, pesticide=pesticide_val, crop_year=int(year), predicted_yield=yield_per_hectare, model_version="xgboost_federated_v1")
            mongo.db.yield_predictions.insert_one(doc)
        except Exception as e:
            print(f"⚠️  Could not save yield prediction: {e}")

        return render_template("YieldPrediction.html", prediction=yield_per_hectare, crops=crops, form_data=request.form, yield_per_acre=yield_per_acre, production=estimated_production, error=None)

    except Exception as e:
        print(f"❌ Yield Error: {e}")
        return render_template("YieldPrediction.html", prediction=None, crops=crops, form_data=request.form, error=f"Prediction error: {str(e)}")


# ╔════════════════════════════════════════════════════════════════════╗
# ║          PEST PREDICTION (with Unknown Detection + Camera)        ║
# ╚════════════════════════════════════════════════════════════════════╝

def _handle_pest_result(save_path, result):
    """Common handler for pest prediction results."""
    pest_name  = result["pest_name"]
    confidence = result["confidence"]
    is_valid   = result["is_valid"]
    message    = result.get("message", "")

    try:
        doc = new_pesticide_rec(
            user_id               = get_current_user_id(),
            image_path            = save_path,
            predicted_pest        = pest_name if is_valid else "unknown",
            recommended_pesticide = pest_name if is_valid else "unknown",
            confidence_score      = float(confidence)
        )
        mongo.db.pesticide_recommendations.insert_one(doc)
    except Exception as e:
        print(f"⚠️  Could not save pesticide recommendation: {e}")

    if is_valid and pest_name != "unknown":
        template_name = pest_name + ".html"
        try:
            return render_template(template_name)
        except Exception:
            return render_template("unaptfile.html")
    else:
        image_path = "/" + save_path.replace("\\", "/")
        return render_template(
            "PestNotRecognized.html",
            image_path=image_path,
            confidence=confidence,
            message=message
        )


@app.route("/predict", methods=["POST"])
@login_required
def pest_predict():
    """Pest prediction from uploaded image."""
    if "image" not in request.files:
        return render_template("unaptfile.html")

    file = request.files["image"]
    if file.filename == "":
        return render_template("unaptfile.html")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if PEST_MODEL_AVAILABLE:
        result = predict_pest(filepath, pest_model)
        return _handle_pest_result(filepath, result)
    else:
        return render_template("unaptfile.html")


@app.route("/predict-camera", methods=["POST"])
@login_required
def pest_predict_camera():
    """Pest prediction from camera capture.
    Camera blob from browser canvas can have RGBA channels,
    wrong EXIF orientation, or non-standard JPEG encoding.
    We re-save through PIL to normalize it before prediction.
    """
    if "image" not in request.files:
        return redirect("/PesticideRecommendation.html")

    file = request.files["image"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"pest_camera_{timestamp}.jpg"
    filepath  = os.path.join(UPLOAD_FOLDER, filename)

    # Sanitize camera image: open with PIL, force RGB, re-save as clean JPEG
    try:
        pil_img = Image.open(file.stream).convert("RGB")
        pil_img.save(filepath, "JPEG", quality=95)
    except Exception as e:
        print(f"⚠️  Camera image save error: {e}")
        file.save(filepath)  # fallback to raw save

    if PEST_MODEL_AVAILABLE:
        result = predict_pest(filepath, pest_model)
        return _handle_pest_result(filepath, result)
    else:
        return render_template("unaptfile.html")


# ╔════════════════════════════════════════════════════════════════════╗
# ║       PLANT DISEASE PREDICTION (with Unknown Detection + Camera)  ║
# ╚════════════════════════════════════════════════════════════════════╝

def _handle_plant_disease_result(save_path, result):
    """Common handler for plant disease prediction results."""
    full_label = result["disease"]
    confidence = result["confidence"]
    is_valid   = result["is_valid"]
    message    = result.get("message", "")

    if is_valid:
        if "___" in full_label:
            plant, disease = full_label.split("___", 1)
        else:
            plant   = full_label
            disease = "Healthy"
        plant_display   = plant.replace("_", " ").title()
        disease_display = disease.replace("_", " ").title()
    else:
        plant_display   = "Unknown"
        disease_display = "Unknown"

    try:
        doc = new_disease_prediction(
            user_id=get_current_user_id(),
            image_path=save_path,
            predicted_disease=full_label if is_valid else "Unknown",
            confidence_score=float(confidence)
        )
        mongo.db.disease_predictions.insert_one(doc)
    except Exception as e:
        print(f"⚠️  Could not save disease prediction: {e}")

    image_path = "/" + save_path.replace("\\", "/")
    return render_template(
        "PlantDiseaseResult.html",
        plant=plant_display,
        disease=disease_display,
        confidence=confidence,
        image_path=image_path,
        is_valid=is_valid,
        message=message
    )


@app.route("/plant-disease-predict", methods=["POST"])
@login_required
def plant_disease_predict():
    if "image" not in request.files:
        return redirect("/PlantDisease.html")
    file = request.files["image"]
    if file.filename == "":
        return redirect("/PlantDisease.html")

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    result = predict_plant_disease(save_path, plant_model, plant_classes)
    return _handle_plant_disease_result(save_path, result)


@app.route("/plant-disease-predict-camera", methods=["POST"])
@login_required
def plant_disease_predict_camera():
    if "image" not in request.files:
        return redirect("/PlantDisease.html")
    file = request.files["image"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"camera_capture_{timestamp}.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    # Sanitize camera image: force RGB, strip EXIF, re-save as clean JPEG
    try:
        pil_img = Image.open(file.stream).convert("RGB")
        pil_img.save(save_path, "JPEG", quality=95)
    except Exception as e:
        print(f"⚠️  Camera image save error: {e}")
        file.save(save_path)

    result = predict_plant_disease(save_path, plant_model, plant_classes)
    return _handle_plant_disease_result(save_path, result)


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    ERROR HANDLERS                                 ║
# ╚════════════════════════════════════════════════════════════════════╝

@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def server_error(error):
    return render_template("error.html", error="Server error"), 500


# ╔════════════════════════════════════════════════════════════════════╗
# ║                    MAIN                                           ║
# ╚════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AGRI AI - COMPLETE SYSTEM WITH AUTHENTICATION")
    print("="*70)

    print("\n📊 SYSTEM STATUS:")
    print(f"  Crop Model:     {'✓ Ready' if crop_recommendation_model else '⚠️  Not loaded'}")
    print(f"  ML Fertilizer:  {'✓ Ready' if FERTILIZER_ML_AVAILABLE else '⚠️  Using Ollama'}")
    print(f"  Yield Model:    {'✓ Ready (98% R²)' if YIELD_MODEL_AVAILABLE else '⚠️  Not loaded'}")
    print(f"  Pest Model:     {'✓ Ready (with unknown detection + camera)' if PEST_MODEL_AVAILABLE else '⚠️  Not loaded — fix numpy/TF versions!'}")
    print(f"  Plant Disease:  {'✓ Ready (with unknown detection + camera)' if plant_model else '⚠️  Not loaded'}")
    print(f"  MongoDB:        ✓ Connected")

    if not PEST_MODEL_AVAILABLE:
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  TO FIX PEST MODEL, run in terminal:                    ║")
        print(f"  ║  pip install numpy==1.24.3 --force-reinstall            ║")
        print(f"  ║  pip install tensorflow==2.15.0 --force-reinstall       ║")
        print(f"  ║  pip install ml-dtypes==0.2.0 --force-reinstall         ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")

    print(f"\n🔐 AUTHENTICATION:")
    print(f"  ✓ User Registration")
    print(f"  ✓ User Login")
    print(f"  ✓ Password Hashing (bcrypt)")
    print(f"  ✓ Session Management")

    print(f"\n💾 DATABASE COLLECTIONS (ERD):")
    print(f"  ✓ users")
    print(f"  ✓ fertilizer_recommendations")
    print(f"  ✓ pesticide_recommendations")
    print(f"  ✓ disease_predictions")
    print(f"  ✓ yield_predictions")
    print(f"  ✓ crop_recommendations")

    print(f"\n🌐 Server: http://127.0.0.1:5000/login")
    print("Press CTRL+C to stop\n")

    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=True)