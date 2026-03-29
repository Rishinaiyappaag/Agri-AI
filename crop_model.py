# train_crop_recommendation.py
# ============================================================
# CROP RECOMMENDATION — STACKING ENSEMBLE + REASONING DATA
# ============================================================
# Also saves crop statistics (min/max/avg for each feature per crop)
# so the result page can show WHY a crop was recommended.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("✓ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  Install XGBoost: pip install xgboost")

# ╔═══════════════════════════════════════════════════════════╗
# ║                    LOAD DATA                              ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("CROP RECOMMENDATION — TRAINING")
print("="*60)

crop = pd.read_csv('Data/crop_recommendation.csv')

print(f"\n📊 Dataset: {len(crop)} samples, {crop['label'].nunique()} crops")

X = crop.iloc[:, :-1].values
Y = crop.iloc[:, -1].values
feature_names = list(crop.columns[:-1])

# ╔═══════════════════════════════════════════════════════════╗
# ║          SAVE CROP STATISTICS (for result page)           ║
# ╚═══════════════════════════════════════════════════════════╝

print("\n📊 Saving crop statistics for result page reasoning...")

crop_stats = {}
for label in crop['label'].unique():
    subset = crop[crop['label'] == label]
    crop_stats[label] = {}
    for feat in feature_names:
        crop_stats[label][feat] = {
            "min": round(float(subset[feat].min()), 1),
            "max": round(float(subset[feat].max()), 1),
            "avg": round(float(subset[feat].mean()), 1)
        }

with open("crop_stats.json", "w") as f:
    json.dump(crop_stats, f, indent=2)
print(f"  ✅ Saved crop_stats.json ({len(crop_stats)} crops)")

# ╔═══════════════════════════════════════════════════════════╗
# ║                    SPLIT & SCALE                          ║
# ╚═══════════════════════════════════════════════════════════╝

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42, stratify=Y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ╔═══════════════════════════════════════════════════════════╗
# ║                    BUILD STACKING MODEL                   ║
# ╚═══════════════════════════════════════════════════════════╝

print(f"\n" + "="*60)
print("BUILDING STACKING ENSEMBLE")
print("="*60)

if XGBOOST_AVAILABLE:
    boost_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, use_label_encoder=False,
        eval_metric='mlogloss', verbosity=0
    )
    print("  ✓ XGBoost (300 trees)")
else:
    boost_model = GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    print("  ✓ GradientBoosting (fallback)")

base_estimators = [
    ('xgb', boost_model),
    ('rf', RandomForestClassifier(
        n_estimators=300, min_samples_split=3,
        max_features='sqrt', random_state=42, n_jobs=-1
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=300, min_samples_split=3,
        random_state=42, n_jobs=-1
    )),
    ('knn', KNeighborsClassifier(
        n_neighbors=5, weights='distance', n_jobs=-1
    )),
]

stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',
    n_jobs=-1,
    passthrough=True
)

# ╔═══════════════════════════════════════════════════════════╗
# ║                    TRAIN                                  ║
# ╚═══════════════════════════════════════════════════════════╝

print(f"\n  Training (1-2 minutes)...")
stacking_model.fit(X_train_scaled, y_train)

train_acc = accuracy_score(y_train, stacking_model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, stacking_model.predict(X_test_scaled))

print(f"\n  Train Accuracy: {train_acc*100:.2f}%")
print(f"  Test Accuracy:  {test_acc*100:.2f}%")

cv_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  CV Accuracy:    {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# Individual model comparison
print(f"\n  Individual models:")
for name, model in base_estimators:
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"    {name:6s}: {acc*100:.2f}%")
print(f"    {'STACK':6s}: {test_acc*100:.2f}% ← BEST")

# ╔═══════════════════════════════════════════════════════════╗
# ║                    SAVE                                   ║
# ╚═══════════════════════════════════════════════════════════╝

pipeline = Pipeline([('scaler', scaler), ('classifier', stacking_model)])
pipeline_acc = accuracy_score(y_test, pipeline.predict(X_test))

with open('Crop_Recommendation.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print(f"\n  ✅ Model saved: Crop_Recommendation.pkl")

# ╔═══════════════════════════════════════════════════════════╗
# ║                    SANITY TESTS                           ║
# ╚═══════════════════════════════════════════════════════════╝

print(f"\n" + "="*60)
print("SANITY TESTS")
print("="*60 + "\n")

with open('Crop_Recommendation.pkl', 'rb') as f:
    loaded = pickle.load(f)

tests = [
    {"name": "Low rainfall 50mm", "input": [40,30,30,35,40,6.5,50],
     "bad": ["coffee","rice","jute","coconut"]},
    {"name": "Low rainfall 200mm", "input": [70,45,40,25,75,6.5,200],
     "bad": []},
    {"name": "Desert 30mm", "input": [10,10,10,38,20,7.5,30],
     "bad": ["rice","coffee","jute","coconut","banana"]},
    {"name": "High rainfall 2500mm", "input": [80,40,40,24,85,5.5,2500],
     "bad": ["mothbeans","lentil","chickpea","muskmelon"]},
    {"name": "Rice conditions", "input": [80,40,40,24,82,5.5,230],
     "bad": ["mothbeans","lentil"]},
]

for t in tests:
    pred = loaded.predict(np.array([t["input"]]))[0]
    fail = pred.lower() in [s.lower() for s in t["bad"]]
    print(f"  {'❌' if fail else '✅'} {t['name']} → {pred}")

# ╔═══════════════════════════════════════════════════════════╗
# ║                    FEATURE IMPORTANCE                     ║
# ╚═══════════════════════════════════════════════════════════╝

print(f"\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60 + "\n")

rf_model = stacking_model.named_estimators_['rf']
importances = rf_model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:14s} {imp:.4f} {bar}")

print(f"\n✅ Done! Restart: python app.py")