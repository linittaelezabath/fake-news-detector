import pandas as pd
import os

DATA_DIR = "data/raw"   # folder with true.csv and fake.csv
OUT_FILE = "data/train.csv"  # final combined file


true_path = os.path.join(DATA_DIR, "true.csv")
fake_path = os.path.join(DATA_DIR, "fake.csv")

df_true = pd.read_csv(true_path)
df_fake = pd.read_csv(fake_path)

df_true["label"] = 0
df_fake["label"] = 1

df = pd.concat([df_true, df_fake], ignore_index=True)

if "title" in df.columns and "text" in df.columns:
    df["text"] = df["title"].astype(str) + " " + df["text"].astype(str)

df = df[["text", "label"]]


df.to_csv(OUT_FILE, index=False)

print(f"Combined dataset saved to {OUT_FILE}")
print(df.head())

