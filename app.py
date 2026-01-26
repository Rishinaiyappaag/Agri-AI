from flask import Flask, render_template, request
from markupsafe import Markup
import os
import numpy as np
import pandas as pd
import pickle

# ================== APP INIT ==================
app = Flask(__name__)

UPLOAD_FOLDER = "static/user_uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== FERTILIZER DATA ==================
from utils.fertilizer import fertilizer_dict

# ================== CROP MODEL ==================
with open("Crop_Recommendation.pkl", "rb") as f:
    crop_recommendation_model = pickle.load(f)

# ================== PEST MODEL ==================
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

pest_model = load_model("Trained_Model.h5")

# ================== PLANT DISEASE MODEL (DENSENET) ==================
from plant_disease_predictor import load_model as load_plant_model
from plant_disease_predictor import predict

PLANT_MODEL_PATH = "plant_disease_densenet.pth"
plant_model, plant_classes = load_plant_model(PLANT_MODEL_PATH)

# ================== PAGE ROUTES ==================
@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/PlantDisease.html")
def plant_disease_page():
    return render_template("PlantDisease.html")

# ================== CROP PREDICTION ==================
@app.route("/crop_prediction", methods=["POST"])
def crop_prediction():
    data = np.array([[
        int(request.form["nitrogen"]),
        int(request.form["phosphorous"]),
        int(request.form["potassium"]),
        float(request.form["temperature"]),
        float(request.form["humidity"]),
        float(request.form["ph"]),
        float(request.form["rainfall"]),
    ]])

    prediction = crop_recommendation_model.predict(data)[0]

    return render_template(
        "crop-result.html",
        prediction=prediction,
        pred="img/crop/" + prediction + ".jpg"
    )

# ================== FERTILIZER PREDICTION ==================
@app.route("/fertilizer-predict", methods=["POST"])
def fertilizer_recommend():
    crop_name = request.form["cropname"]
    N_filled = int(request.form["nitrogen"])
    P_filled = int(request.form["phosphorous"])
    K_filled = int(request.form["potassium"])

    df = pd.read_csv("Data/Crop_NPK.csv")
    row = df[df["Crop"] == crop_name].iloc[0]

    def check(diff):
        if diff < 0:
            return "High"
        elif diff > 0:
            return "low"
        else:
            return "No"

    n = check(row["N"] - N_filled)
    p = check(row["P"] - P_filled)
    k = check(row["K"] - K_filled)

    return render_template(
        "Fertilizer-Result.html",
        recommendation1=Markup(fertilizer_dict["N" + n]),
        recommendation2=Markup(fertilizer_dict["P" + p]),
        recommendation3=Markup(fertilizer_dict["K" + k]),
    )

# ================== PEST HELPER ==================
def predict_pest(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    preds = pest_model.predict(img)
    return np.argmax(preds)

# ================== PEST PREDICTION ==================
@app.route("/predict", methods=["POST"])
def pest_predict():
    if "image" not in request.files:
        return render_template("unaptfile.html")

    file = request.files["image"]
    if file.filename == "":
        return render_template("unaptfile.html")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    pest_index = predict_pest(filepath)

    pest_map = {
        0: "aphids",
        1: "armyworm",
        2: "beetle",
        3: "bollworm",
        4: "earthworm",
        5: "grasshopper",
        6: "mites",
        7: "mosquito",
        8: "sawfly",
        9: "stem borer",
    }

    return render_template(pest_map.get(pest_index, "unaptfile") + ".html")

# ================== PLANT DISEASE PREDICTION ==================
@app.route("/plant-disease-predict", methods=["POST"])
def plant_disease_predict():
    if "image" not in request.files:
        return render_template("unaptfile.html")

    file = request.files["image"]
    if file.filename == "":
        return render_template("unaptfile.html")

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    result = predict(save_path, plant_model, plant_classes)

    # Expected label: tomato___healthy
    full_label = result["disease"]
    confidence = result["confidence"]

    if "___" in full_label:
        plant, disease = full_label.split("___")
    else:
        plant = full_label
        disease = "Unknown"

    image_path = "/" + save_path.replace("\\", "/")

    return render_template(
        "PlantDiseaseResult.html",
        plant=plant.title(),
        disease=disease.replace("_", " ").title(),
        confidence=confidence,
        image_path=image_path
    )

# ================== MAIN ==================
if __name__ == "__main__":
    app.run(debug=True)
