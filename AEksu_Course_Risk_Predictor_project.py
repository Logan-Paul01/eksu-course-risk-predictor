
import pandas as pd
from pandas import read_csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ==========================
# LOAD DATA
# ==========================

data = read_csv('/storage/emulated/0/eksu_course_risk_dataset.csv')

print(data.head())
print(data.info())
print(data.describe())

# ==========================
# SPLIT FEATURES & TARGET
# ==========================

# 0 = pass, 1 = fail
X = data.drop(['risk_label'], axis=1)
y = data['risk_label']

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y   # VERY IMPORTANT for classification
)

# ==========================
# MODEL INITIALIZATION
# ==========================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

pipeline = Pipeline([
    ("model", model)
])

# ==========================
# TRAIN MODEL
# ==========================

pipeline.fit(X_train, y_train)

# ==========================
# PREDICTIONS
# ==========================

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of FAIL (class 1)

print("Sample Failure Probabilities:")
print(y_prob[:10])

# ==========================
# EVALUATION
# ==========================

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# ==========================
# FEATURE IMPORTANCE
# ==========================

# Extract trained model from pipeline
trained_model = pipeline.named_steps["model"]

importance = trained_model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:\n", feature_importance)