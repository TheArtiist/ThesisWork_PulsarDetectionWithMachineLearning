import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_plots():
    output_dir = "statisztikai_abrak"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    column_names = [
        "ip_mean", "ip_std", "ip_kurtosis", "ip_skewness",
        "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness",
        "label"
    ]

    df = pd.read_csv("htru2/HTRU_2.csv", header=None, names=column_names)
    
    df.dropna(inplace=True)
    df = df.select_dtypes(include=[np.number])
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df = df[indices_to_keep]

    features = [col for col in column_names if col != "label"]

    plt.figure(figsize=(8, 6))
    counts = df["label"].value_counts()
    plt.bar([str(x) for x in counts.index], counts.values, color=['blue', 'orange'])
    plt.title("Osztályok eloszlása (0: Nem pulzár, 1: Pulzár)")
    plt.xlabel("Osztály")
    plt.ylabel("Darabszám")
    plt.savefig(os.path.join(output_dir, "osztalyeloszlas.png"))
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr()
    cax = plt.matshow(corr, cmap='coolwarm')
    plt.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='left')
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    for (i, j), z in np.ndenumerate(corr):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                 
    plt.title("Jellemzők korrelációs mátrixa", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "korrelacios_matrix.png"))
    plt.close()

    for feature in features:
        plt.figure(figsize=(8, 6))
        plt.hist([df[df['label'] == 0][feature], df[df['label'] == 1][feature]], 
                 bins=50, stacked=False, label=['0', '1'], color=['blue', 'orange'], alpha=0.7)
        plt.title(f"{feature} hisztogramja osztályonként")
        plt.xlabel(feature)
        plt.ylabel("Gyakoriság")
        plt.legend(title='label')
        plt.savefig(os.path.join(output_dir, f"hist_{feature}.png"))
        plt.close()

    for feature in features:
        plt.figure(figsize=(8, 6))
        plt.boxplot([df[df['label'] == 0][feature], df[df['label'] == 1][feature]], labels=['0', '1'])
        plt.title(f"{feature} boxplotja osztályonként")
        plt.xlabel("Osztály")
        plt.ylabel(feature)
        plt.savefig(os.path.join(output_dir, f"boxplot_{feature}.png"))
        plt.close()

    print(f"Az összes ábra sikeresen mentve a(z) '{output_dir}' mappába.")

if __name__ == "__main__":
    generate_plots()
