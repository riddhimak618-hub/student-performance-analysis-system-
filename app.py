import pickle
import numpy as np

# Load saved model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("=== Student Performance Prediction System ===")
print("Enter student details to predict Pass or Fail\n")

# Take input from user
study_hours = float(input("Enter Study Hours (e.g. 6): "))
attendance = float(input("Enter Attendance Percentage (e.g. 75): "))
previous_scores = float(input("Enter Previous Scores (e.g. 70): "))

# Prepare input
student = np.array([[study_hours, attendance, previous_scores]])
student_scaled = scaler.transform(student)

# Predict
prediction = model.predict(student_scaled)
probability = model.predict_proba(student_scaled)

result = "PASS" if prediction[0] == 1 else "FAIL"

print("\n--- Prediction Result ---")
print(f"Study Hours    : {study_hours}")
print(f"Attendance     : {attendance}%")
print(f"Previous Score : {previous_scores}")
print(f"Result         : {result}")
print(f"Pass Probability : {probability[0][1]*100:.2f}%")
print(f"Fail Probability : {probability[0][0]*100:.2f}%")
