import pandas as pd
import os

DATA_DIR = "data/raw"   # folder with true.csv and fake.csv
OUT_FILE = "data/train.csv"  # final combined file


true_path = os.path.join(DATA_DIR, "true.csv")
fake_path = os.path.join(DATA_DIR, "fake.csv")

# Read both
df_true = pd.read_csv(true_path)
df_fake = pd.read_csv(fake_path)

# Assume both have a column "text" or "content" — adjust if different
# Add labels: 0 = REAL, 1 = FAKE
df_true["label"] = 0
df_fake["label"] = 1

# Concatenate
df = pd.concat([df_true, df_fake], ignore_index=True)

# If there's a "title" column and a "text" column, merge them
if "title" in df.columns and "text" in df.columns:
    df["text"] = df["title"].astype(str) + " " + df["text"].astype(str)

# Keep only needed columns
df = df[["text", "label"]]

# Save combined file
df.to_csv(OUT_FILE, index=False)

print(f"✅ Combined dataset saved to {OUT_FILE}")
print(df.head())
