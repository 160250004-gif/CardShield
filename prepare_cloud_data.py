import pandas as pd
from pathlib import Path

source = Path("creditcard.csv")
destination = Path("creditcard_sample.csv")

if not source.exists():
    raise FileNotFoundError("creditcard.csv nahi mili.")

df = pd.read_csv(source)

fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0].sample(
    n=min(10000, int((df["Class"] == 0).sum())),
    random_state=42
)

sample = pd.concat([normal, fraud], ignore_index=True)
sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_csv(destination, index=False)

print("Sample dataset created:", destination)
print("Total rows:", len(sample))
print("Fraud rows:", int(sample["Class"].sum()))
