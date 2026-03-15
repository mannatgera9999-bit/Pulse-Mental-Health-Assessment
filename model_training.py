import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------------------
# 1. Load Dataset
# -------------------------------------------
DATA_PATH = os.path.join("data", "mental_health_data.csv")
df = pd.read_csv(DATA_PATH)

# FEATURES in your CSV:
feature_cols = [
    "age",
    "gender",
    "sleep_hours",
    "work_hours",
    "screen_time",
    "physical_activity",
    "anxiety_score",
    "depression_score",
    "mood_swings",
    "family_history",
]

target_col = "stress_level"
X = df[feature_cols]
y = df[target_col].astype(str)

# -------------------------------------------
# 2. Train-Test Split
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------
# 3. Preprocessor
# -------------------------------------------
numeric_features = [
    "age", "sleep_hours", "work_hours", "screen_time",
    "physical_activity", "anxiety_score", "depression_score"
]

categorical_features = ["gender", "mood_swings", "family_history"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------------------------
# 4. Classification Model - Random Forest
# -------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", rf_clf),
])

clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------
# 5. Clustering Model (k-Means)
# -------------------------------------------
X_processed_full = preprocessor.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_processed_full)

# -------------------------------------------
# 6. Save Models
# -------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(clf_pipeline, "models/stress_rf_pipeline.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\nModels successfully saved to the 'models/' folder!")
