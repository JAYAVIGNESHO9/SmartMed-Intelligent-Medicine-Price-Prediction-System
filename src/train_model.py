import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from math import sqrt


DATA_PATH = r"D:\Project\learnML\Smart Medicine Price & Availability Predictor\data\processed_medicines.pkl"
df = pd.read_pickle(DATA_PATH)

print("✅ Processed data loaded successfully")
print("Dataset shape:", df.shape)

X = df.drop("price_inr", axis=1)
y = df["price_inr"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model training completed")


y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))  # RMSE
r2 = r2_score(y_test, y_pred)

print(f"💡 Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# =========================================
# 6️⃣ Save the trained model
# =========================================
MODEL_PATH = r"D:\Project\learnML\Smart Medicine Price & Availability Predictor\data\medicine_price_model.pkl"
joblib.dump(model, MODEL_PATH)

print("✅ Trained model saved to:", MODEL_PATH)
