import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("data/pnl_risk_data.csv")

X=df.drop("ReconBreak",axis=1)

y=df["ReconBreak"]

model=RandomForestClassifier(
n_estimators=100,
random_state=42
)

model.fit(X,y)

joblib.dump(
model,
"models/break_model.pkl"
)

print("Model Saved")
