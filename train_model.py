import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
df = pd.read_csv("student_data.csv")

# Separate features and target
X = df[['study_hours', 'attendance', 'previous_scores']]
y = df['result']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluation
print("Model: Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail', 'Pass'],
            yticklabels=['Fail', 'Pass'])
plt.title("Confusion Matrix - Student Performance")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix saved as confusion_matrix.png")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")

# Test with a new student
print("\n--- Predict for a New Student ---")
import numpy as np
new_student = np.array([[6, 78, 70]])   # study_hours, attendance, previous_scores
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)
probability = model.predict_proba(new_student_scaled)

result = "PASS" if prediction[0] == 1 else "FAIL"
print(f"Study Hours: 6 | Attendance: 78% | Previous Score: 70")
print(f"Prediction  : {result}")
print(f"Probability : Pass={probability[0][1]:.2f}, Fail={probability[0][0]:.2f}")
