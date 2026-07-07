import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv", header=None)

# FEATURES
X = df.iloc[:, :-1]

# LABELS
y = df.iloc[:, -1]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- SCALER ----------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

# ---------------- TRAIN ----------------
model.fit(X_train, y_train)

# ---------------- TEST ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.4f}")

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

# ---------------- SAVE ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\n🎯 Model + Scaler saved successfully!")