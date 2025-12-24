"""
Automatic AI/ML + Graph Visualizer for Beginners

What this program does:
1. Asks for a CSV file path.
2. Automatically:
   - Loads the CSV
   - Drops rows with missing values
   - Uses the LAST column as the target (y)
   - Uses all numeric columns (except target) as features (X)
   - Detects whether it's a classification or regression problem
   - Trains a Decision Tree model
   - Prints sample predictions and a metric (accuracy or MSE)
3. Opens a matplotlib window with:
   - 2D scatter plots for some column pairs
   - 3D scatter plot using the first 3 numeric columns (if available)
"""

import itertools

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


def plot_graphs(X, y=None):
    """
    Show 2D and 3D graphs based on numeric features.

    - 2D: scatter plots for up to 6 pairs of numeric columns
    - 3D: one scatter plot using the first 3 numeric columns (if available)
    """
    numeric_cols = list(X.columns)
    n = len(numeric_cols)

    if n == 0:
        print("\n(No numeric columns available for plotting.)")
        return

    # Simple color mapping for classes (if y is given)
    color_values = None
    if y is not None:
        # Convert classes to integers for coloring
        color_values = pd.factorize(y)[0]

    # ---------- 2D Scatter Plots ----------
    if n >= 2:
        pairs = list(itertools.combinations(range(n), 2))
        max_plots = min(len(pairs), 6)  # limit to avoid too many subplots

        print(f"\nCreating {max_plots} 2D scatter plots...")

        fig, axes = plt.subplots(1, max_plots, figsize=(5 * max_plots, 4))
        if max_plots == 1:
            axes = [axes]  # make it iterable

        for ax, (i, j) in zip(axes, pairs[:max_plots]):
            xname = numeric_cols[i]
            yname = numeric_cols[j]

            if color_values is not None:
                ax.scatter(X[xname], X[yname], c=color_values, alpha=0.7)
            else:
                ax.scatter(X[xname], X[yname], alpha=0.7)

            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            ax.set_title(f"{xname} vs {yname}")

        fig.suptitle("2D Scatter Plots (numeric feature pairs)", fontsize=14)

    # ---------- 3D Scatter Plot ----------
    if n >= 3:
        print("Creating 3D scatter plot...")

        fig3 = plt.figure(figsize=(6, 5))
        ax3 = fig3.add_subplot(111, projection="3d")

        a, b, c = numeric_cols[:3]

        if color_values is not None:
            ax3.scatter(X[a], X[b], X[c], c=color_values, alpha=0.7)
        else:
            ax3.scatter(X[a], X[b], X[c], alpha=0.7)

        ax3.set_xlabel(a)
        ax3.set_ylabel(b)
        ax3.set_zlabel(c)
        ax3.set_title("3D Scatter Plot (first 3 numeric features)")

    # Show all figures at once
    plt.show()


def main():
    # 1. Ask user for CSV path
    csv_path = input("Enter the path to your CSV file: ").strip()

    # 2. Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Error reading CSV:", e)
        raise SystemExit

    print("\n✅ CSV loaded successfully!")
    print(f"Shape of data: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nFirst 5 rows:\n")
    print(df.head())

    print("\nColumns:\n")
    print(list(df.columns))

    # 3. Drop rows with missing values
    df = df.dropna()
    print(f"\nAfter dropping rows with missing values: {df.shape[0]} rows remain")

    if df.shape[0] == 0:
        print("❌ No data left after removing missing values. Please clean your CSV.")
        raise SystemExit

    # 4. Auto-select target: LAST column
    target_col = df.columns[-1]
    print(f"\n🎯 Auto-selected TARGET column: '{target_col}'")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 5. Keep only numeric feature columns
    X = X.select_dtypes(include=["int64", "float64"])

    if X.empty:
        print("\n❌ No numeric feature columns found to train on.")
        print("Hint: Add some numeric columns (e.g., counts, amounts, measurements).")
        raise SystemExit

    print("\nUsing these feature columns:\n", list(X.columns))

    # 6. Decide classification vs regression
    is_classification = (y.dtype == "O") or (y.nunique() <= 20)
    problem_type = "classification" if is_classification else "regression"
    print(f"\nProblem type detected: {problem_type}")

    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 8. Train and evaluate model
    if is_classification:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("\n🔍 Sample predictions vs actual (first 10):")
        print("Predictions:", list(y_pred[:10]))
        print("Actual     :", list(y_test[:10].values))
        print(f"\n✅ Accuracy: {acc * 100:.2f}%")

    else:
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print("\n🔍 Sample predictions vs actual (first 10):")
        print("Predictions:", [round(v, 2) for v in y_pred[:10]])
        print("Actual     :", list(y_test[:10].values))
        print(f"\n✅ Mean Squared Error: {mse:.2f}")

    # 9. Plot graphs on a canvas
    plot_graphs(X, y)


if __name__ == "__main__":
    main()
