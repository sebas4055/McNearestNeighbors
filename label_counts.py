import os
import pandas as pd
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")

data = pd.read_csv(CSV_PATH)
n = len(data)

print(f"Total entries in train.csv: {n}")

label_counts = data['TARGET'].value_counts().sort_index()

print("\nFrequency of each label:")
print(label_counts)

plt.figure(figsize=(20,20))  
label_counts.plot(kind='barh', color='skyblue', edgecolor='black')

plt.title('Distribution of Labels in train.csv', fontsize=16)
plt.gca().invert_yaxis() 
plt.savefig('label_distribution_horizontal.png')

