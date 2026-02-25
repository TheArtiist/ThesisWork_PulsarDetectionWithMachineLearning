import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True) 
    df = df.select_dtypes(include=[np.number])  
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep]

def split_dataset():
    column_names = [
        "ip_mean", "ip_std", "ip_kurtosis", "ip_skewness",
        "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness",
        "label"
    ]
    
    print("Célfájl (HTRU_2.csv) betöltése...")
    dataset = pd.read_csv("htru2/HTRU_2.csv", header=None, names=column_names)
    dataFrame = clean_dataset(dataset)
    
    print("Adathalmaz szétválasztása (70% tanító, 15% validációs, 15% teszt)...")
    train_df, temp_df = train_test_split(dataFrame, test_size=0.30, random_state=42, stratify=dataFrame["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])
    
    train_file = "htru2/train.csv"
    val_file = "htru2/validation.csv"
    test_file = "htru2/test.csv"
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Tanító halmaz mentve: {train_file} (Méret: {len(train_df)})")
    print(f"Validációs halmaz mentve: {val_file} (Méret: {len(val_df)})")
    print(f"Teszt halmaz mentve: {test_file} (Méret: {len(test_df)})")

if __name__ == "__main__":
    split_dataset()
