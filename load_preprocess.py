import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------
df = pd.read_csv("student_data.csv")

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:", df.shape)

print("\nDataset Info:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nPass/Fail Count:")
print(df['result'].value_counts())

# Pairplot
sns.pairplot(df, hue='result', palette={0: 'red', 1: 'green'})
plt.suptitle("Student Performance - Pairplot", y=1.02)
plt.savefig("pairplot.png", bbox_inches='tight')
plt.show()
print("Pairplot saved as pairplot.png")

# -----------------------------------------------
# PREPROCESS DATA
# -----------------------------------------------
X = df[['study_hours', 'attendance', 'previous_scores']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nTraining samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])
print("Features used    :", list(X.columns))
print("\nPreprocessing complete. Scaler saved as scaler.pkl")
