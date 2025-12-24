"""
Elementary AI/ML program for beginners using YOUR CSV file.

Requirements:
    pip install pandas scikit-learn

How to prepare your CSV:
    - One row = one data sample
    - Each column (except the target) = a feature (number is best: int/float)
    - One column = target/label (e.g., "species", "passed", "category", etc.)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Ask user for CSV path
csv_path = input("Enter the path to your CSV file: ").strip()

# 2. Load the CSV
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print("Error reading CSV:", e)
    exit(1)

print("\n CSV loaded successfully!")
print("\nHere are the first 5 rows of your data:\n")
print(df.head())

print("\nThese are your columns:\n")
print(list(df.columns))

# 3. Ask user which column is the target/label
target_col = input(
    "\nType the name of the TARGET column (the thing you want to predict): "
).strip()

if target_col not in df.columns:
    print(f"\n❌ Column '{target_col}' not found in the CSV.")
    exit(1)

# 4. Create features (X) and labels (y)
X = df.drop(columns=[target_col])  # all columns except target
y = df[target_col]                 # only target column

# Optional: Try to keep only numeric columns in X (good for beginners)
X = X.select_dtypes(include=["int64", "float64"])

if X.empty:
    print("\n❌ No numeric feature columns found to train on.")
    print("Make sure you have some numeric columns for features.")
    exit(1)

print("\nUsing these feature columns:\n", list(X.columns))

# 5. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 6. Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 7. Predict on test data
y_pred = model.predict(X_test)

# 8. Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n🔍 Sample predictions vs actual:")
print("Predictions:", list(y_pred[:10]))
print("Actual     :", list(y_test[:10].values))

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
