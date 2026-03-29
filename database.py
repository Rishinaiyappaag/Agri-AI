"""
database.py - MongoDB Database Setup for AGRI AI
Creates all collections with schema validation as per the ERD diagram.

Collections:
  - users
  - fertilizer_recommendations
  - pesticide_recommendations
  - disease_predictions
  - yield_predictions
  - crop_recommendations

FIX: All numeric fields now accept both "int" and "double" bsonTypes
because Python int → MongoDB int32, Python float → MongoDB double.
The old validator only accepted "double" which silently rejected
documents where values like nitrogen=50 (int) were passed.
"""

from pymongo import MongoClient, ASCENDING
from pymongo.errors import CollectionInvalid
from datetime import datetime


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

def get_db(uri="mongodb://localhost:27017/", db_name="agri_ai_db"):
    """Connect to MongoDB and return the database object."""
    client = MongoClient(uri)
    return client[db_name]


# ─────────────────────────────────────────────
# SCHEMA VALIDATORS  (one per collection)
# ─────────────────────────────────────────────
# Using ["int", "double"] for all numeric fields so both
# Python int and float values are accepted by MongoDB.

USERS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "email", "password_hash", "role", "created_at"],
        "properties": {
            "name":          {"bsonType": "string",  "description": "Full name of the user"},
            "email":         {"bsonType": "string",  "description": "Unique email address"},
            "password_hash": {"bsonType": "string",  "description": "Bcrypt hashed password"},
            "role":          {"bsonType": "string",  "enum": ["farmer", "admin"],
                              "description": "User role"},
            "created_at":    {"bsonType": "date",    "description": "Account creation timestamp"},
        }
    }
}

FERTILIZER_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "nitrogen", "phosphorus", "potassium",
                     "crop_type", "recommended_fertilizer", "created_at"],
        "properties": {
            "user_id":                {"bsonType": "objectId"},
            "nitrogen":               {"bsonType": ["int", "double"]},
            "phosphorus":             {"bsonType": ["int", "double"]},
            "potassium":              {"bsonType": ["int", "double"]},
            "crop_type":              {"bsonType": "string"},
            "recommended_fertilizer": {"bsonType": "string"},
            "created_at":             {"bsonType": "date"},
        }
    }
}

PESTICIDE_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "image_path", "predicted_pest",
                     "recommended_pesticide", "confidence_score", "created_at"],
        "properties": {
            "user_id":               {"bsonType": "objectId"},
            "image_path":            {"bsonType": "string"},
            "predicted_pest":        {"bsonType": "string"},
            "recommended_pesticide": {"bsonType": "string"},
            "confidence_score":      {"bsonType": ["int", "double"]},
            "created_at":            {"bsonType": "date"},
        }
    }
}

DISEASE_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "image_path", "predicted_disease",
                     "confidence_score", "created_at"],
        "properties": {
            "user_id":           {"bsonType": "objectId"},
            "image_path":        {"bsonType": "string"},
            "predicted_disease": {"bsonType": "string"},
            "confidence_score":  {"bsonType": ["int", "double"]},
            "created_at":        {"bsonType": "date"},
        }
    }
}

YIELD_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "area", "annual_rainfall", "fertilizer",
                     "pesticide", "crop_year", "predicted_yield",
                     "model_version", "created_at"],
        "properties": {
            "user_id":         {"bsonType": "objectId"},
            "area":            {"bsonType": ["int", "double"]},
            "annual_rainfall": {"bsonType": ["int", "double"]},
            "fertilizer":      {"bsonType": ["int", "double"]},
            "pesticide":       {"bsonType": ["int", "double"]},
            "crop_year":       {"bsonType": ["int", "double"]},
            "predicted_yield": {"bsonType": ["int", "double"]},
            "model_version":   {"bsonType": "string"},
            "created_at":      {"bsonType": "date"},
        }
    }
}

CROP_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "nitrogen", "phosphorus", "potassium",
                     "soil_ph", "rainfall", "temperature", "humidity",
                     "recommended_crop", "created_at"],
        "properties": {
            "user_id":          {"bsonType": "objectId"},
            "nitrogen":         {"bsonType": ["int", "double"]},
            "phosphorus":       {"bsonType": ["int", "double"]},
            "potassium":        {"bsonType": ["int", "double"]},
            "soil_ph":          {"bsonType": ["int", "double"]},
            "rainfall":         {"bsonType": ["int", "double"]},
            "temperature":      {"bsonType": ["int", "double"]},
            "humidity":         {"bsonType": ["int", "double"]},
            "recommended_crop": {"bsonType": "string"},
            "created_at":       {"bsonType": "date"},
        }
    }
}


# ─────────────────────────────────────────────
# COLLECTION DEFINITIONS
# ─────────────────────────────────────────────

COLLECTIONS = [
    {
        "name":      "users",
        "validator": USERS_VALIDATOR,
        "indexes": [
            {"keys": [("email", ASCENDING)], "unique": True, "name": "idx_users_email"},
        ]
    },
    {
        "name":      "fertilizer_recommendations",
        "validator": FERTILIZER_VALIDATOR,
        "indexes": [
            {"keys": [("user_id", ASCENDING)], "name": "idx_fertilizer_user_id"},
        ]
    },
    {
        "name":      "pesticide_recommendations",
        "validator": PESTICIDE_VALIDATOR,
        "indexes": [
            {"keys": [("user_id", ASCENDING)], "name": "idx_pesticide_user_id"},
        ]
    },
    {
        "name":      "disease_predictions",
        "validator": DISEASE_VALIDATOR,
        "indexes": [
            {"keys": [("user_id", ASCENDING)], "name": "idx_disease_user_id"},
        ]
    },
    {
        "name":      "yield_predictions",
        "validator": YIELD_VALIDATOR,
        "indexes": [
            {"keys": [("user_id", ASCENDING)], "name": "idx_yield_user_id"},
        ]
    },
    {
        "name":      "crop_recommendations",
        "validator": CROP_VALIDATOR,
        "indexes": [
            {"keys": [("user_id", ASCENDING)], "name": "idx_crop_user_id"},
        ]
    },
]


# ─────────────────────────────────────────────
# INITIALISER
# ─────────────────────────────────────────────

def init_db(uri="mongodb://localhost:27017/", db_name="agri_ai_db"):
    """
    Create all collections (with validators) and indexes.
    Safe to call on an already-initialised database — existing
    collections are left untouched and only missing indexes are added.
    """
    db = get_db(uri, db_name)

    for col_def in COLLECTIONS:
        col_name  = col_def["name"]
        validator = col_def["validator"]
        indexes   = col_def.get("indexes", [])

        try:
            db.create_collection(
                col_name,
                validator=validator,
                validationLevel="moderate",
                validationAction="error"
            )
            print(f"  ✅ Created collection: {col_name}")
        except CollectionInvalid:
            # Collection already exists — update validator
            db.command("collMod", col_name,
                       validator=validator,
                       validationLevel="moderate",
                       validationAction="error")
            print(f"  ♻️  Updated validator:  {col_name}")

        col = db[col_name]
        for idx in indexes:
            col.create_index(idx["keys"],
                             unique=idx.get("unique", False),
                             name=idx["name"])
            print(f"     └─ index: {idx['name']}")

    print("\n✅ Database initialised successfully.")
    return db


# ─────────────────────────────────────────────
# DOCUMENT HELPERS  (one per collection)
# ─────────────────────────────────────────────

def new_fertilizer_rec(user_id, nitrogen, phosphorus, potassium,
                       crop_type, recommended_fertilizer):
    """Fertilizer_Recommendations document."""
    return {
        "user_id":                user_id,
        "nitrogen":               float(nitrogen),
        "phosphorus":             float(phosphorus),
        "potassium":              float(potassium),
        "crop_type":              str(crop_type),
        "recommended_fertilizer": str(recommended_fertilizer),
        "created_at":             datetime.utcnow(),
    }


def new_pesticide_rec(user_id, image_path, predicted_pest,
                      recommended_pesticide, confidence_score):
    """Pesticide_Recommendations document."""
    return {
        "user_id":               user_id,
        "image_path":            str(image_path),
        "predicted_pest":        str(predicted_pest),
        "recommended_pesticide": str(recommended_pesticide),
        "confidence_score":      float(confidence_score),
        "created_at":            datetime.utcnow(),
    }


def new_disease_prediction(user_id, image_path, predicted_disease, confidence_score):
    """Disease_Predictions document."""
    return {
        "user_id":           user_id,
        "image_path":        str(image_path),
        "predicted_disease": str(predicted_disease),
        "confidence_score":  float(confidence_score),
        "created_at":        datetime.utcnow(),
    }


def new_yield_prediction(user_id, area, annual_rainfall, fertilizer,
                         pesticide, crop_year, predicted_yield, model_version):
    """Yield_Predictions document."""
    return {
        "user_id":         user_id,
        "area":            float(area),
        "annual_rainfall": float(annual_rainfall),
        "fertilizer":      float(fertilizer),
        "pesticide":       float(pesticide),
        "crop_year":       int(crop_year),
        "predicted_yield": float(predicted_yield),
        "model_version":   str(model_version),
        "created_at":      datetime.utcnow(),
    }


def new_crop_recommendation(user_id, nitrogen, phosphorus, potassium,
                            soil_ph, rainfall, temperature, humidity,
                            recommended_crop):
    """Crop_Recommendations document."""
    return {
        "user_id":          user_id,
        "nitrogen":         float(nitrogen),
        "phosphorus":       float(phosphorus),
        "potassium":        float(potassium),
        "soil_ph":          float(soil_ph),
        "rainfall":         float(rainfall),
        "temperature":      float(temperature),
        "humidity":         float(humidity),
        "recommended_crop": str(recommended_crop),
        "created_at":       datetime.utcnow(),
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Initialising AGRI AI database...\n")
    init_db()