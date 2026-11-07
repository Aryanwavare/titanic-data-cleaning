import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Work relative to this script's folder so it runs from any terminal
base_dir = os.path.dirname(os.path.abspath(__file__))

# ---------- Part 1: Load and summarize ----------
df = pd.read_csv(os.path.join(base_dir, "train.csv"))

print("First 5 rows:")
print(df.head())

print("\n--- .info() ---")
df.info()

print("\n--- .describe() ---")
print(df.describe())

# ---------- Part 2: Handle missing data ----------
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Median for Age (skewed, so median is safer than mean)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Mode for Embarked (categorical, only 2 values missing)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin is mostly missing, so we drop it rather than guess
df = df.drop(columns=["Cabin"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ---------- Part 3: Encode categorical variables ----------
# LabelEncoder on Sex (binary: female/male -> 0/1)
le = LabelEncoder()
df["Sex_label"] = le.fit_transform(df["Sex"])

# OneHotEncoder on Embarked (3 ports -> 3 binary columns)
ohe = OneHotEncoder(sparse_output=False)
embarked_encoded = ohe.fit_transform(df[["Embarked"]])
embarked_cols = ohe.get_feature_names_out(["Embarked"])
df = pd.concat(
    [df, pd.DataFrame(embarked_encoded, columns=embarked_cols, index=df.index)],
    axis=1
)

print("\nSex encoded with LabelEncoder (first 5):")
print(df[["Sex", "Sex_label"]].head())

print("\nEmbarked encoded with OneHotEncoder (first 5):")
print(df[["Embarked"] + list(embarked_cols)].head())

# ---------- Part 4: Visualize age distribution ----------
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="steelblue")
plt.title("Age Distribution (after imputation)")
plt.xlabel("Age")
plt.ylabel("Number of passengers")
plt.tight_layout()
plt.show()

# ---------- Part 5: Save cleaned dataset ----------
output_path = os.path.join(base_dir, "titanic_cleaned.csv")
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")
print("Final shape:", df.shape)