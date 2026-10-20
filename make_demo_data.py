import pandas as pd

df = pd.read_csv("creditcard.csv")

# Sample transactions for app demonstration
demo = df.drop(columns=["Class"]).sample(
    n=100,
    random_state=42
)

demo.to_csv("demo_transactions.csv", index=False)

print("Demo CSV created successfully!")