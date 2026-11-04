import pandas as pd
import os

file_path = os.path.abspath("data/raw/dataset.csv")

print("Checking file:", file_path)

if not os.path.exists(file_path):
    print("ERROR: File does not exist")
else:
    print("File size:", os.path.getsize(file_path), "bytes")

    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read(500)

    print("\nFile content preview:")
    print(repr(content))

    if not content.strip():
        print("ERROR: CSV is empty or contains only blank spaces")
    else:
        df = pd.read_csv(file_path)
        print("\nSUCCESS: Dataset loaded!")
        print("Shape:", df.shape)
        print(df.head())