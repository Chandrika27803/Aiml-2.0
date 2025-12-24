"""
AI/ML + Graph Dashboard (Tabbed 2D / 3D, with plot suggestions)

- Asks for CSV path (via console input)
- Auto:
  - Loads CSV
  - Drops rows with missing values
  - Uses LAST column as target (y)
  - Uses all numeric columns (except target) as features (X)
  - Detects classification vs regression
  - Trains Decision Tree model and prints performance
- Analyzes columns:
  - Detects numeric / categorical / datetime columns
  - Prints intelligent 2D/3D plot suggestions (like suggest_plots.py)
- Opens a Tkinter window with:
  - Tab 1: 2D graphs (scatter, line, hist, box, categorical vs numeric, time series)
  - Tab 2: 3D graphs (3D scatter if possible)
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


# ---------- Plot suggestion helpers (analyst-style) ----------

def analyze_columns(df: pd.DataFrame):
    """
    Classify columns into numeric, categorical, and datetime.
    Tries to parse object columns as datetime when possible.
    """
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in df.columns:
        series = df[col]

        # Already datetime?
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        # Try parse as datetime if object-like
        if series.dtype == "object":
            try:
                parsed = pd.to_datetime(series, errors="raise")
                df[col] = parsed
                datetime_cols.append(col)
                continue
            except Exception:
                pass

        # Now detect numeric vs categorical
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, datetime_cols


def build_plot_suggestions_text(
    numeric_cols,
    categorical_cols,
    datetime_cols
) -> str:
    """
    Generate human-readable 2D/3D plot suggestions as a big string,
    similar in spirit to suggest_plots.py.
    """
    lines = []

    def add(line=""):
        print(line)
        lines.append(line)

    add("Detected column types:")
    add(f"  Numeric     : {numeric_cols}")
    add(f"  Categorical : {categorical_cols}")
    add(f"  Datetime    : {datetime_cols}")

    # --- 2D suggestions ---
    add("\n=== Possible 2D Graphs ===")

    if not numeric_cols:
        add("No numeric columns found → very limited plotting options.")
    else:
        add("\n1) Single numeric column plots:")
        for col in numeric_cols:
            add(f"   - Line plot: {col} vs row index")
            add(f"   - Histogram: distribution of {col}")
            add(f"   - Box plot: distribution of {col}")

        if len(numeric_cols) >= 2:
            add("\n2) Numeric vs numeric plots (pairs of numeric columns):")
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    x = numeric_cols[i]
                    y = numeric_cols[j]
                    add(f"   - Scatter plot: {x} vs {y}")
                    add(f"   - Line plot: {y} vs {x}")

        if categorical_cols:
            add("\n3) Categorical vs numeric plots:")
            for cat in categorical_cols:
                for num in numeric_cols:
                    add(f"   - Bar plot: average {num} per {cat}")
                    add(f"   - Box plot: {num} grouped by {cat}")

        if datetime_cols:
            add("\n4) Time series plots (datetime vs numeric):")
            for dt in datetime_cols:
                for num in numeric_cols:
                    add(f"   - Line plot: {num} vs {dt}")
                    add(f"   - Scatter plot: {num} vs {dt}")

    # --- 3D suggestions ---
    add("\n=== Possible 3D Graphs ===")

    if len(numeric_cols) < 3:
        add("Fewer than 3 numeric columns → limited 3D options.")
    else:
        add("\n1) 3D scatter plots (triplets of numeric columns):")
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                for k in range(j + 1, len(numeric_cols)):
                    x = numeric_cols[i]
                    y = numeric_cols[j]
                    z = numeric_cols[k]
                    add(f"   - 3D scatter: {x} vs {y} vs {z}")

    if datetime_cols and len(numeric_cols) >= 2:
        add("\n2) 3D plots with time axis (datetime + 2 numeric):")
        for dt in datetime_cols:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    x = dt
                    y = numeric_cols[i]
                    z = numeric_cols[j]
                    add(f"   - 3D line/scatter: {x} (time) vs {y} vs {z}")

    return "\n".join(lines)


# ---------- GUI Helpers ----------

class ScrollableFrame(ttk.Frame):
    """
    A vertically scrollable frame for Tkinter.
    We embed the matplotlib canvases inside this frame.
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


# ---------- Figure creators (2D & 3D) ----------

def _sample_df(df, max_rows=5000):
    """Sample the dataframe for plotting (to avoid choking on huge CSVs)."""
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df


def create_scatter_figure(X, y=None, max_pairs=6):
    """Multiple 2D scatter plots for numeric feature pairs."""
    numeric_cols = list(X.columns)
    n = len(numeric_cols)
    if n < 2:
        return None

    X_plot = _sample_df(X)

    pairs = list(itertools.combinations(range(n), 2))[:max_pairs]
    n_plots = len(pairs)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    color_values = None
    if y is not None:
        # Align y with X_plot index
        y_plot = y.loc[X_plot.index]
        color_values = pd.factorize(y_plot)[0]

    for ax, (i, j) in zip(axes, pairs):
        xname = numeric_cols[i]
        yname = numeric_cols[j]

        if color_values is not None:
            ax.scatter(X_plot[xname], X_plot[yname], c=color_values, alpha=0.7)
        else:
            ax.scatter(X_plot[xname], X_plot[yname], alpha=0.7)

        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(f"{xname} vs {yname}")

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("2D Scatter Plots (numeric feature pairs)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_line_figure(X):
    """Line plots: index vs each numeric feature."""
    numeric_cols = list(X.columns)
    n = len(numeric_cols)
    if n == 0:
        return None

    X_plot = _sample_df(X)

    n_cols = 2 if n > 1 else 1
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        ax.plot(X_plot.index, X_plot[col])
        ax.set_xlabel("Row index (sampled)")
        ax.set_ylabel(col)
        ax.set_title(f"Line Plot: {col}")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Line Plots (Index vs Feature)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_hist_figure(X):
    """Histograms for each numeric feature."""
    numeric_cols = list(X.columns)
    n = len(numeric_cols)
    if n == 0:
        return None

    X_plot = _sample_df(X)

    n_cols = 2 if n > 1 else 1
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        ax.hist(X_plot[col], bins=20, alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram: {col}")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Histograms (Numeric Features)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_box_figure(X):
    """Boxplot for all numeric columns."""
    numeric_cols = list(X.columns)
    if len(numeric_cols) == 0:
        return None

    X_plot = _sample_df(X)

    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols)), 4))
    ax.boxplot(
        [X_plot[col].dropna() for col in numeric_cols],
        labels=numeric_cols,
        vert=True
    )
    ax.set_title("Boxplot (Numeric Features)")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig


def create_categorical_bar_figure(df, categorical_cols, numeric_cols):
    """Bar plot: average of a numeric column per category (first cat/num)."""
    if not categorical_cols or not numeric_cols:
        return None

    df_plot = _sample_df(df)
    cat = categorical_cols[0]
    num = numeric_cols[0]

    # Take top categories by count to avoid insane bars
    counts = df_plot[cat].value_counts().head(10).index
    grp = df_plot[df_plot[cat].isin(counts)].groupby(cat)[num].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    grp.plot(kind="bar", ax=ax)
    ax.set_xlabel(cat)
    ax.set_ylabel(f"Average {num}")
    ax.set_title(f"Bar Plot: Average {num} per {cat}")
    fig.tight_layout()
    return fig


def create_time_series_figure(df, datetime_cols, numeric_cols):
    """Time series: first numeric vs first datetime."""
    if not datetime_cols or not numeric_cols:
        return None

    dt_col = datetime_cols[0]
    num_col = numeric_cols[0]

    # Filter out NaT
    df_plot = df[[dt_col, num_col]].dropna()
    df_plot = _sample_df(df_plot).sort_values(dt_col)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_plot[dt_col], df_plot[num_col])
    ax.set_xlabel(dt_col)
    ax.set_ylabel(num_col)
    ax.set_title(f"Time Series: {num_col} vs {dt_col}")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def create_3d_figure(X, y=None):
    """
    Create a matplotlib Figure with a single 3D scatter plot using
    the first 3 numeric columns.
    """
    numeric_cols = list(X.columns)
    if len(numeric_cols) < 3:
        return None

    X_plot = _sample_df(X, max_rows=2000)

    a, b, c = numeric_cols[:3]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    color_values = None
    if y is not None:
        y_plot = y.loc[X_plot.index]
        color_values = pd.factorize(y_plot)[0]

    if color_values is not None:
        ax.scatter(X_plot[a], X_plot[b], X_plot[c], c=color_values, alpha=0.7)
    else:
        ax.scatter(X_plot[a], X_plot[b], X_plot[c], alpha=0.7)

    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_zlabel(c)
    ax.set_title("3D Scatter Plot (first 3 numeric features)")

    fig.tight_layout()
    return fig


def open_dashboard(df_clean, X_numeric, y, numeric_cols, categorical_cols, datetime_cols):
    """
    Build the Tkinter dashboard window with tabbed 2D/3D graphs.
    """
    root = tk.Tk()
    root.title("AI/ML Graph Dashboard")
    root.geometry("1200x700")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # ----- 2D TAB -----
    tab2d = ScrollableFrame(notebook)
    notebook.add(tab2d, text="2D Graphs")

    # Scatter
    fig_scatter = create_scatter_figure(X_numeric, y)
    if fig_scatter is not None:
        canvas_scatter = FigureCanvasTkAgg(fig_scatter, master=tab2d.scrollable_frame)
        canvas_scatter.draw()
        canvas_scatter.get_tk_widget().pack(fill="both", expand=True, pady=5)
    else:
        ttk.Label(tab2d.scrollable_frame, text="Not enough numeric columns for 2D scatter plots").pack(pady=10)

    # Line plots
    fig_line = create_line_figure(X_numeric)
    if fig_line is not None:
        canvas_line = FigureCanvasTkAgg(fig_line, master=tab2d.scrollable_frame)
        canvas_line.draw()
        canvas_line.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Histograms
    fig_hist = create_hist_figure(X_numeric)
    if fig_hist is not None:
        canvas_hist = FigureCanvasTkAgg(fig_hist, master=tab2d.scrollable_frame)
        canvas_hist.draw()
        canvas_hist.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Boxplots
    fig_box = create_box_figure(X_numeric)
    if fig_box is not None:
        canvas_box = FigureCanvasTkAgg(fig_box, master=tab2d.scrollable_frame)
        canvas_box.draw()
        canvas_box.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Categorical vs numeric (bar)
    fig_cat = create_categorical_bar_figure(df_clean, categorical_cols, numeric_cols)
    if fig_cat is not None:
        canvas_cat = FigureCanvasTkAgg(fig_cat, master=tab2d.scrollable_frame)
        canvas_cat.draw()
        canvas_cat.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Time series (datetime vs numeric)
    fig_ts = create_time_series_figure(df_clean, datetime_cols, numeric_cols)
    if fig_ts is not None:
        canvas_ts = FigureCanvasTkAgg(fig_ts, master=tab2d.scrollable_frame)
        canvas_ts.draw()
        canvas_ts.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # ----- 3D TAB -----
    tab3d = ScrollableFrame(notebook)
    notebook.add(tab3d, text="3D Graphs")

    fig3d = create_3d_figure(X_numeric, y)
    if fig3d is not None:
        canvas3d = FigureCanvasTkAgg(fig3d, master=tab3d.scrollable_frame)
        canvas3d.draw()
        canvas3d.get_tk_widget().pack(fill="both", expand=True, pady=20)
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

    # 2.5 Analyze columns BEFORE dropping rows, so we see full structure
    # (df may be modified in-place to parse datetimes)
    df_for_analysis = df.copy()
    numeric_cols_all, categorical_cols, datetime_cols = analyze_columns(df_for_analysis)

    # Build and print suggestions
    suggestions_text = build_plot_suggestions_text(
        numeric_cols_all, categorical_cols, datetime_cols
    )

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
    X_full = df.drop(columns=[target_col])

    # 5. Keep only numeric feature columns for ML
    X_numeric = X_full.select_dtypes(include=["int64", "float64"])

    if X_numeric.empty:
        print("\n❌ No numeric feature columns found to train on.")
        print("Hint: Add some numeric columns (e.g., counts, amounts, measurements).")
        # Still open dashboard with what we have (might be limited)
        df_clean = df.copy()
        open_dashboard(df_clean, X_numeric, y, numeric_cols_all, categorical_cols, datetime_cols)
        raise SystemExit

    print("\nUsing these feature columns for ML:\n", list(X_numeric.columns))

    # 6. Decide classification vs regression (basic heuristic)
    n_unique = y.nunique()
    is_classification = (y.dtype == "O") or (n_unique <= 20)
    problem_type = "classification" if is_classification else "regression"
    print(f"\nProblem type detected: {problem_type} (unique target values: {n_unique})")

    # Guard: avoid insane multi-class classification with huge unique labels
    if is_classification and n_unique > 50:
        print(f"\n⚠ Target '{target_col}' has {n_unique} unique values.")
        print("Too many classes for this simple demo. Skipping model training, showing graphs only.")
        df_clean = df.copy()
        open_dashboard(df_clean, X_numeric, y, numeric_cols_all, categorical_cols, datetime_cols)
        return

    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.3, random_state=42
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
    df_clean = df_for_analysis  # with parsed datetimes where possible
    open_dashboard(df_clean, X_numeric, y, numeric_cols_all, categorical_cols, datetime_cols)


if __name__ == "__main__":
    main()
