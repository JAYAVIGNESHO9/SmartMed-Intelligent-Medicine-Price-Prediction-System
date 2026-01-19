import pandas as pd
import joblib

MODEL_PATH = r"D:\Project\learnML\Smart Medicine Price & Availability Predictor\data\medicine_price_model.pkl"
model = joblib.load(MODEL_PATH)


DATA_PATH = r"D:\Project\learnML\Smart Medicine Price & Availability Predictor\data\processed_medicines.pkl"
df = pd.read_pickle(DATA_PATH)

feature_cols = df.drop("price_inr", axis=1).columns.tolist()
print("✅ Model and feature structure loaded")
print(f"Expected features: {feature_cols}")


def predict_price(user_input: dict):
    """
    user_input: dict with feature names and values
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # One-hot encode categorical features (must match training)
    input_df = pd.get_dummies(input_df)

    # Align with training features
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # Predict
    price = model.predict(input_df)[0]
    return round(price, 2)


if __name__ == "__main__":

    print("\n💊 Smart Medicine Price Predictor 💊\n")

    # Example interactive input
    brand_name = input("Enter brand name: ")
    manufacturer = input("Enter manufacturer: ")
    dosage_form = input("Enter dosage form (Tablet/Capsule/Injection etc.): ")
    pack_size = int(input("Enter pack size (number of units): "))
    pack_unit = input("Enter pack unit (Tablet/ml/etc.): ")
    num_active_ingredients = int(input("Enter number of active ingredients: "))
    primary_ingredient = input("Enter primary ingredient (salt): ")
    strength_mg = float(input("Enter strength in mg: "))
    therapeutic_class = input("Enter therapeutic class: ")
    is_discontinued = int(input("Is discontinued? (0 = No, 1 = Yes): "))

    # Create input dictionary
    user_input = {
        "brand_name": brand_name,
        "manufacturer": manufacturer,
        "dosage_form": dosage_form,
        "pack_size": pack_size,
        "pack_unit": pack_unit,
        "num_active_ingredients": num_active_ingredients,
        "primary_ingredient": primary_ingredient,
        "strength_mg": strength_mg,
        "therapeutic_class": therapeutic_class,
        "is_discontinued": is_discontinued
    }

    # Predict price
    predicted_price = predict_price(user_input)
    print(f"\n💰 Predicted Medicine Price: ₹{predicted_price}")
