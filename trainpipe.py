import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("generated/pose_train.csv")

X = df.drop("target", axis=1)
y = df["target"]

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

model = GradientBoostingClassifier(n_estimators=300)
model.fit(Xtr, ytr)

print(classification_report(yte, model.predict(Xte)))

joblib.dump(model, "outcome_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")