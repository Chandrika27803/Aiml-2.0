"""
AI/ML + Graph Dashboard (Tabbed 2D / 3D)

- Asks for CSV path (via console input)
- Auto:
  - Loads CSV
  - Drops rows with missing values
  - Uses LAST column as target (y)
  - Uses all numeric columns (except target) as features (X)
  - Detects classification vs regression
  - Trains Decision Tree model and prints performance
- Opens a Tkinter window with:
  - Tab 1: 2D scatter plots (scrollable)
  - Tab 2: 3D scatter plot (scrollable) if possible
"""

import itertools

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

import matplotlib
matplotlib.use("TkAgg")  # ensure Tk backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------- GUI Helpers ----------

class ScrollableFrame(ttk.Frame):
    """
    A vertically scrollable frame for Tkinter.
    We embed the matplotlib canvas inside this frame.
    """
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas
        self.scrollable_frame = scrollable_frame


def create_2d_figure(X, y=None, max_plots=6):
    """
    Create a matplotlib Figure with multiple 2D scatter plots.
    """
    numeric_cols = list(X.columns)
    n = len(numeric_cols)

    if n < 2:
        return None  # not enough columns

    pairs = list(itertools.combinations(range(n), 2))
    pairs = pairs[:max_plots]

    # Decide grid layout: at most 2 columns
    n_plots = len(pairs)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color mapping if y is provided
    color_values = None
    if y is not None:
        color_values = pd.factorize(y)[0]

    for ax, (i, j) in zip(axes, pairs):
        xname = numeric_cols[i]
        yname = numeric_cols[j]

        if color_values is not None:
            ax.scatter(X[xname], X[yname], c=color_values, alpha=0.7)
        else:
            ax.scatter(X[xname], X[yname], alpha=0.7)

        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(f"{xname} vs {yname}")

    # If there are unused axes (when plots < rows*cols), hide them
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("2D Scatter Plots (numeric feature pairs)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_3d_figure(X, y=None):
    """
    Create a matplotlib Figure with a single 3D scatter plot using
    the first 3 numeric columns.
    """
    numeric_cols = list(X.columns)
    if len(numeric_cols) < 3:
        return None  # not enough columns

    a, b, c = numeric_cols[:3]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    color_values = None
    if y is not None:
        color_values = pd.factorize(y)[0]

    if color_values is not None:
        ax.scatter(X[a], X[b], X[c], c=color_values, alpha=0.7)
    else:
        ax.scatter(X[a], X[b], X[c], alpha=0.7)

    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_zlabel(c)
    ax.set_title("3D Scatter Plot (first 3 numeric features)")

    fig.tight_layout()
    return fig


def open_dashboard(X, y):
    """
    Build the Tkinter dashboard window with tabbed 2D/3D graphs.
    """
    root = tk.Tk()
    root.title("AI/ML Graph Dashboard")

    # Make window reasonably large
    root.geometry("1200x700")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # 2D tab
    tab2d = ScrollableFrame(notebook)
    notebook.add(tab2d, text="2D Graphs")

    fig2d = create_2d_figure(X, y)
    if fig2d is not None:
        canvas2d = FigureCanvasTkAgg(fig2d, master=tab2d.scrollable_frame)
        canvas2d.draw()
        canvas2d.get_tk_widget().pack(fill="both", expand=True)
    else:
        ttk.Label(tab2d.scrollable_frame, text="Not enough numeric columns for 2D plots").pack(pady=20)

    # 3D tab
    tab3d = ScrollableFrame(notebook)
    notebook.add(tab3d, text="3D Graphs")

    fig3d = create_3d_figure(X, y)
    if fig3d is not None:
        canvas3d = FigureCanvasTkAgg(fig3d, master=tab3d.scrollable_frame)
        canvas3d.draw()
        canvas3d.get_tk_widget().pack(fill="both", expand=True)
    else:
        ttk.Label(tab3d.scrollable_frame, text="Not enough numeric columns for 3D plot").pack(pady=20)

    root.mainloop()


# ---------- Main AI/ML Logic ----------

def main():
    # 1. Ask user for CSV path (console)
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

    # 9. Open the GUI dashboard with tabs for 2D and 3D plots
    open_dashboard(X, y)


if __name__ == "__main__":
    main()
