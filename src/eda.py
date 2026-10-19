
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed.cleveland.data"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

df = pd.read_csv(DATA, names=columns, na_values="?")
for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["target"] = (df["num"] > 0).astype(int)
X = df.drop(columns=["num", "target"])
y = df["target"]

df.describe(include="all").to_csv(OUT / "eda_summary.csv")
X.isna().sum().rename("missing_count").to_csv(OUT / "missing_values.csv")
y.value_counts().sort_index().rename_axis("target").to_csv(
    OUT / "class_distribution.csv"
)

plt.figure()
y.value_counts().sort_index().plot(kind="bar")
plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = absent, 1 = present)")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUT / "target_distribution.png", dpi=150)
plt.close()

plt.figure()
df["age"].dropna().plot(kind="hist", bins=15)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(OUT / "age_distribution.png", dpi=150)
plt.close()

print("Rows:", len(df))
print("Features:", X.shape[1])
print("Missing values by feature:")
print(X.isna().sum())
print("EDA outputs saved to:", OUT)
