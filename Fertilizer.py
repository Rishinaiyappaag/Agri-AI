import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ================= LOAD DATA =================

data = pd.read_csv(r"C:\MCA FINAL PROJECT\Agri AI\Data\Fertilizer Prediction.csv")

# 🔥 CLEAN COLUMN NAMES
data.columns = data.columns.str.strip()

print("Columns:", data.columns)

# ================= ENCODE CATEGORICAL =================

soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

data["Soil Type"] = soil_encoder.fit_transform(data["Soil Type"])
data["Crop Type"] = crop_encoder.fit_transform(data["Crop Type"])
data["Fertilizer Name"] = fertilizer_encoder.fit_transform(data["Fertilizer Name"])

# ================= FEATURES & TARGET =================

X = data[[
    "Temparature",
    "Humidity",
    "Moisture",
    "Soil Type",
    "Crop Type",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
]]

y = data["Fertilizer Name"]

# ================= SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= STACKING =================

base_models = [
    ("rf", RandomForestClassifier(n_estimators=200)),
    ("svm", SVC(probability=True)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
]

meta_model = LogisticRegression(max_iter=1000)

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

stack_model.fit(X_train, y_train)

# ================= ACCURACY =================

y_pred = stack_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Stacking Accuracy:", accuracy)

# ================= SAVE =================

pickle.dump(stack_model, open("Fertilizer_Stack_Model.pkl", "wb"))
pickle.dump(soil_encoder, open("soil_encoder.pkl", "wb"))
pickle.dump(crop_encoder, open("crop_encoder.pkl", "wb"))
pickle.dump(fertilizer_encoder, open("fertilizer_encoder.pkl", "wb"))

print("Model saved successfully!")
