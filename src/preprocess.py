import pandas as pd
import numpy as np


DATA_PATH = r"D:\Project\learnML\Smart Medicine Price & Availability Predictor\data\indian_pharmaceutical_products.csv"

df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())


selected_cols = [
    'brand_name',
    'manufacturer',
    'dosage_form',
    'pack_size',
    'pack_unit',
    'num_active_ingredients',
    'primary_ingredient',
    'primary_strength',
    'therapeutic_class',
    'price_inr',
    'is_discontinued'
]

# keep only existing columns
selected_cols = [col for col in selected_cols if col in df.columns]
df = df[selected_cols]

print("Selected columns:", df.columns.tolist())


# numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# boolean column
if 'is_discontinued' in df.columns:
    df['is_discontinued'] = df['is_discontinued'].fillna(False).astype(int)

# ===============================
# 4. Feature engineering
# ===============================
# Extract numeric strength (e.g. "500 mg" → 500)
if 'primary_strength' in df.columns:
    df['strength_mg'] = (
        df['primary_strength']
        .astype(str)
        .str.extract(r'(\d+\.?\d*)')[0]
        .astype(float)
    )
    df['strength_mg'] = df['strength_mg'].fillna(df['strength_mg'].median())
    df.drop(columns=['primary_strength'], inplace=True)

# ===============================
# 5. Encode categorical features
# ===============================
df = pd.get_dummies(
    df,
    columns=cat_cols,
    drop_first=True
)

print("Shape after encoding:", df.shape)

# ===============================
# 6. Remove invalid prices
# ===============================
df = df[df['price_inr'] > 0]

# ===============================
# 7. Save processed data (FAST & SAFE)
# ===============================
OUTPUT_PATH = r"D:\Project\learnML\Smart Medicine Price & Availability Predictor\data\processed_medicines.pkl"

df.to_pickle(OUTPUT_PATH)

print("✅ Preprocessing completed successfully")
print("Final shape:", df.shape)
print("Saved to:", OUTPUT_PATH)
