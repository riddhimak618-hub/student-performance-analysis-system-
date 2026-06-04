# Student Performance Prediction System

A machine learning project that predicts whether a student will **Pass or Fail** based on:
- Study Hours
- Attendance Percentage
- Previous Scores

## Project Structure

```
student-performance-prediction/
    student_data.csv          # Dataset
    step2_load.py             # Load and explore data
    step3_preprocess.py       # Preprocess and scale data
    step4_train_model.py      # Train and evaluate ML model
    step5_push_to_github.sh   # Git push script
    model.pkl                 # Saved trained model
    scaler.pkl                # Saved scaler
    confusion_matrix.png      # Model evaluation chart
    pairplot.png              # Data visualization
    README.md
```

## Steps to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Load and explore data
```bash
python step2_load.py
```

### 3. Preprocess data
```bash
python step3_preprocess.py
```

### 4. Train the model
```bash
python step4_train_model.py
```

## Model Used
- **Logistic Regression** (from scikit-learn)

## Dataset
- 20 student records with features: study_hours, attendance, previous_scores
- Target: result (1 = Pass, 0 = Fail)

## Results
The model predicts student pass/fail with accuracy printed after running step4_train_model.py.
