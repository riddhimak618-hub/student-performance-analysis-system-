import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("StudentsPerformance.csv")
print("Data loaded!")
print(df.head())

df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
df['result'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)
df['study_hours'] = np.random.randint(1, 10, size=len(df))

le = LabelEncoder()
for col in ['gender','race/ethnicity','parental level of education','lunch','test preparation course']:
    df[col] = le.fit_transform(df[col])

print("Preprocessing done!")

X = df[['gender','race/ethnicity','parental level of education','lunch','test preparation course','study_hours']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, 'student_model.pkl')
print("Model saved!")

plt.figure(figsize=(8,5))
sns.countplot(x='result', data=df)
plt.title('Pass vs Fail Count')
plt.xticks([0,1], ['Fail','Pass'])
plt.savefig('result_chart.png')
plt.show()
print("Chart saved!")
