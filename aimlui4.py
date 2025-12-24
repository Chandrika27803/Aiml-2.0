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
  - Prints intelligent 2D/3D plot suggestions

- Opens a Tkinter window with:
  - Tab 1: Possible Graphs (text suggestions)
  - Tab 2: 2D graphs (scatter, line, hist, box, cat vs num, time series)
  - Tab 3: 3D graphs (numeric 3D, time+2 numeric)

- Closing the window exits the program.
"""

import itertools
import sys

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


# --------- Global limits so suggestions == actually plotted combos ---------

MAX_NUMERIC_PAIRS_2D = 6      # how many numeric-vs-numeric pairs
MAX_PLOT_SAMPLE_ROWS = 5000   # max rows for plotting


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
    Generate human-readable 2D/3D plot suggestions as a big string.

    IMPORTANT: Only suggests the same combinations we actually draw,
    so the Possible Graphs tab matches the graphs.
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
    add("\n=== 2D Graphs (examples actually drawn) ===")

    if not numeric_cols:
        add("No numeric columns found -> very limited plotting options.")
    else:
        # 1) Single numeric: line vs index, hist, box
        add("\n1) Single numeric column plots (drawn for each numeric column):")
        for col in numeric_cols:
            add(f"   - Line vs index: {col} vs row index")
            add(f"   - Histogram: distribution of {col}")
            add(f"   - Box plot: distribution of {col}")

        # 2) Numeric vs numeric (limited pairs)
        pairs = list(itertools.combinations(numeric_cols, 2))[:MAX_NUMERIC_PAIRS_2D]
        if pairs:
            add("\n2) Numeric vs numeric plots (limited pairs, all drawn):")
            for x, y in pairs:
                add(f"   - Scatter: {x} vs {y}")
                add(f"   - Line: {y} vs {x}")

        # 3) Categorical vs numeric (only first cat+num, matching plots)
        if categorical_cols and numeric_cols:
            cat = categorical_cols[0]
            num = numeric_cols[0]
            add("\n3) Categorical vs numeric plots (one example pair drawn):")
            add(f"   - Bar: average {num} per {cat}")
            add(f"   - Box: {num} grouped by {cat}")

        # 4) Time series (only first dt+num, matching plots)
        if datetime_cols and numeric_cols:
            dt = datetime_cols[0]
            num = numeric_cols[0]
            add("\n4) Time series plots (one example pair drawn):")
            add(f"   - Line: {num} vs {dt}")
            add(f"   - Scatter: {num} vs {dt}")

    # --- 3D suggestions ---
    add("\n=== 3D Graphs (examples actually drawn) ===")

    # Numeric 3D: only first triplet (matches create_3d_figure)
    if len(numeric_cols) >= 3:
        a, b, c = numeric_cols[:3]
        add("\n1) 3D scatter with numeric triplet:")
        add(f"   - 3D scatter: {a} vs {b} vs {c}")
    else:
        add("Not enough numeric columns for 3D numeric scatter (need at least 3).")

    # Time + 2 numeric: only first dt + first two nums (matches create_3d_time_figure)
    if datetime_cols and len(numeric_cols) >= 2:
        dt = datetime_cols[0]
        y = numeric_cols[0]
        z = numeric_cols[1]
        add("\n2) 3D with time axis:")
        add(f"   - 3D scatter: {dt} (time) vs {y} vs {z}")
    else:
        add("\nNo suitable combination for 3D time-based scatter (need 1 datetime + 2 numeric).")

    return "\n".join(lines)


# ---------- GUI Helpers ----------

class ScrollableFrame(ttk.Frame):
    """
    A vertically scrollable frame for Tkinter.
    We embed the matplotlib canvases inside this frame.

    The inner frame auto-expands horizontally to match the window width.
    """
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", _on_frame_configure)

        # Create window for inner frame and keep the ID so we can resize it
        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_canvas_configure(event):
            # Make inner frame width track canvas width
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas
        self.scrollable_frame = scrollable_frame


# ---------- Figure creators (2D & 3D) ----------

def _sample_df(df, max_rows=MAX_PLOT_SAMPLE_ROWS):
    """Sample the dataframe for plotting (to avoid choking on huge CSVs)."""
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df


def create_scatter_figure(X, y=None):
    """Multiple 2D scatter plots for numeric feature pairs (limited)."""
    numeric_cols = list(X.columns)
    n = len(numeric_cols)
    if n < 2:
        return None

    X_plot = _sample_df(X)

    pairs_idx = list(itertools.combinations(range(n), 2))[:MAX_NUMERIC_PAIRS_2D]
    n_plots = len(pairs_idx)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    color_values = None
    if y is not None:
        y_plot = y.loc[X_plot.index]
        color_values = pd.factorize(y_plot)[0]

    for ax, (i, j) in zip(axes, pairs_idx):
        xname = numeric_cols[i]
        yname = numeric_cols[j]

        if color_values is not None:
            ax.scatter(X_plot[xname], X_plot[yname], c=color_values, alpha=0.7)
        else:
            ax.scatter(X_plot[xname], X_plot[yname], alpha=0.7)

        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(f"Scatter: {xname} vs {yname}")

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("2D Scatter Plots (numeric feature pairs)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_pair_line_figure(X):
    """
    Line plots: y vs x for numeric pairs (limited),
    matching the 'Line: y vs x' suggestions.
    """
    numeric_cols = list(X.columns)
    n = len(numeric_cols)
    if n < 2:
        return None

    X_plot = _sample_df(X)

    pairs_idx = list(itertools.combinations(range(n), 2))[:MAX_NUMERIC_PAIRS_2D]
    n_plots = len(pairs_idx)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (i, j) in zip(axes, pairs_idx):
        xname = numeric_cols[i]
        yname = numeric_cols[j]

        data = X_plot[[xname, yname]].dropna().sort_values(xname)
        if data.empty:
            ax.set_visible(False)
            continue

        ax.plot(data[xname], data[yname])
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(f"Line: {yname} vs {xname}")

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle("Line Plots (numeric vs numeric)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_line_index_figure(X):
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
        ax.set_title(f"Line vs Index: {col}")

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
        tick_labels=numeric_cols,  # updated for Matplotlib 3.9+
        vert=True
    )
    ax.set_title("Boxplot (Numeric Features)")
    ax.set_ylabel("Value")
    fig.tight_layout()
    return fig


def create_box_by_category_figure(df, categorical_cols, numeric_cols):
    """Box plot: numeric grouped by category (first cat/num only)."""
    if not categorical_cols or not numeric_cols:
        return None

    cat = categorical_cols[0]
    num = numeric_cols[0]

    df_plot = _sample_df(df[[cat, num]].dropna())

    groups = []
    labels = []
    for value, grp in df_plot.groupby(cat):
        groups.append(grp[num].values)
        labels.append(str(value))
        if len(labels) >= 10:  # limit categories
            break

    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), 4))
    ax.boxplot(groups, tick_labels=labels)  # updated for Matplotlib 3.9+
    ax.set_xlabel(cat)
    ax.set_ylabel(num)
    ax.set_title(f"Boxplot: {num} grouped by {cat}")
    fig.tight_layout()
    return fig


def create_categorical_bar_figure(df, categorical_cols, numeric_cols):
    """Bar plot: average of a numeric column per category (first cat/num)."""
    if not categorical_cols or not numeric_cols:
        return None

    df_plot = _sample_df(df)
    cat = categorical_cols[0]
    num = numeric_cols[0]

    counts = df_plot[cat].value_counts().head(10).index
    grp = df_plot[df_plot[cat].isin(counts)].groupby(cat)[num].mean()

    if grp.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    grp.plot(kind="bar", ax=ax)
    ax.set_xlabel(cat)
    ax.set_ylabel(f"Average {num}")
    ax.set_title(f"Bar Plot: Average {num} per {cat}")
    fig.tight_layout()
    return fig


def create_time_series_line_figure(df, datetime_cols, numeric_cols):
    """Time series: line plot num vs datetime (first dt/num)."""
    if not datetime_cols or not numeric_cols:
        return None

    dt_col = datetime_cols[0]
    num_col = numeric_cols[0]

    df_plot = df[[dt_col, num_col]].dropna()
    df_plot = _sample_df(df_plot).sort_values(dt_col)

    if df_plot.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_plot[dt_col], df_plot[num_col])
    ax.set_xlabel(dt_col)
    ax.set_ylabel(num_col)
    ax.set_title(f"Time Series (Line): {num_col} vs {dt_col}")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def create_time_series_scatter_figure(df, datetime_cols, numeric_cols):
    """Time series: scatter num vs datetime (first dt/num)."""
    if not datetime_cols or not numeric_cols:
        return None

    dt_col = datetime_cols[0]
    num_col = numeric_cols[0]

    df_plot = df[[dt_col, num_col]].dropna()
    df_plot = _sample_df(df_plot).sort_values(dt_col)

    if df_plot.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df_plot[dt_col], df_plot[num_col], alpha=0.7)
    ax.set_xlabel(dt_col)
    ax.set_ylabel(num_col)
    ax.set_title(f"Time Series (Scatter): {num_col} vs {dt_col}")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def create_3d_figure(X, y=None):
    """3D scatter plot using first 3 numeric columns."""
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
    ax.set_title("3D Scatter (first 3 numeric features)")

    fig.tight_layout()
    return fig


def create_3d_time_figure(df, datetime_cols, numeric_cols):
    """
    3D scatter: time (x) + 2 numeric (y,z) – first dt + first 2 numerics.
    Matches '3D line/scatter: time vs num1 vs num2' suggestion.
    """
    if not datetime_cols or len(numeric_cols) < 2:
        return None

    dt_col = datetime_cols[0]
    y_col = numeric_cols[0]
    z_col = numeric_cols[1]

    df_plot = df[[dt_col, y_col, z_col]].dropna()
    df_plot = _sample_df(df_plot, max_rows=2000).sort_values(dt_col)

    if df_plot.empty:
        return None

    # For 3D axis, convert time to numeric (seconds since epoch)
    t_numeric = df_plot[dt_col].astype("int64") // 10**9

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(t_numeric, df_plot[y_col], df_plot[z_col], alpha=0.7)

    ax.set_xlabel(f"{dt_col} (timestamp)")
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f"3D Scatter: time vs {y_col} vs {z_col}")

    fig.tight_layout()
    return fig


def open_dashboard(df_clean, X_numeric, y,
                   numeric_cols_all, categorical_cols, datetime_cols,
                   suggestions_text: str):
    """
    Build the Tkinter dashboard window with tabbed Possible Graphs / 2D / 3D.
    Closing the window ends the program.
    """
    root = tk.Tk()
    root.title("AI/ML Graph Dashboard")
    root.geometry("1400x800")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # ----- Possible Graphs TAB (text first, as requested) -----
    tab_sugg = ScrollableFrame(notebook)
    notebook.add(tab_sugg, text="Possible Graphs")

    text_widget = tk.Text(tab_sugg.scrollable_frame, wrap="word")
    text_widget.insert("1.0", suggestions_text)
    text_widget.config(state="disabled")
    text_widget.pack(fill="both", expand=True, padx=5, pady=5)

    # ----- 2D TAB -----
    tab2d = ScrollableFrame(notebook)
    notebook.add(tab2d, text="2D Graphs")

    # Scatter (numeric vs numeric)
    fig_scatter = create_scatter_figure(X_numeric, y)
    if fig_scatter is not None:
        canvas_scatter = FigureCanvasTkAgg(fig_scatter, master=tab2d.scrollable_frame)
        canvas_scatter.draw()
        canvas_scatter.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Line (y vs x for pairs)
    fig_pair_line = create_pair_line_figure(X_numeric)
    if fig_pair_line is not None:
        canvas_pair_line = FigureCanvasTkAgg(fig_pair_line, master=tab2d.scrollable_frame)
        canvas_pair_line.draw()
        canvas_pair_line.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Line vs index
    fig_line_index = create_line_index_figure(X_numeric)
    if fig_line_index is not None:
        canvas_line_index = FigureCanvasTkAgg(fig_line_index, master=tab2d.scrollable_frame)
        canvas_line_index.draw()
        canvas_line_index.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Histograms
    fig_hist = create_hist_figure(X_numeric)
    if fig_hist is not None:
        canvas_hist = FigureCanvasTkAgg(fig_hist, master=tab2d.scrollable_frame)
        canvas_hist.draw()
        canvas_hist.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Boxplots (numeric only)
    fig_box = create_box_figure(X_numeric)
    if fig_box is not None:
        canvas_box = FigureCanvasTkAgg(fig_box, master=tab2d.scrollable_frame)
        canvas_box.draw()
        canvas_box.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Categorical vs numeric: bar & box
    fig_cat_bar = create_categorical_bar_figure(df_clean, categorical_cols, numeric_cols_all)
    if fig_cat_bar is not None:
        canvas_cat_bar = FigureCanvasTkAgg(fig_cat_bar, master=tab2d.scrollable_frame)
        canvas_cat_bar.draw()
        canvas_cat_bar.get_tk_widget().pack(fill="both", expand=True, pady=5)

    fig_cat_box = create_box_by_category_figure(df_clean, categorical_cols, numeric_cols_all)
    if fig_cat_box is not None:
        canvas_cat_box = FigureCanvasTkAgg(fig_cat_box, master=tab2d.scrollable_frame)
        canvas_cat_box.draw()
        canvas_cat_box.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # Time series: line & scatter
    fig_ts_line = create_time_series_line_figure(df_clean, datetime_cols, numeric_cols_all)
    if fig_ts_line is not None:
        canvas_ts_line = FigureCanvasTkAgg(fig_ts_line, master=tab2d.scrollable_frame)
        canvas_ts_line.draw()
        canvas_ts_line.get_tk_widget().pack(fill="both", expand=True, pady=5)

    fig_ts_scatter = create_time_series_scatter_figure(df_clean, datetime_cols, numeric_cols_all)
    if fig_ts_scatter is not None:
        canvas_ts_scatter = FigureCanvasTkAgg(fig_ts_scatter, master=tab2d.scrollable_frame)
        canvas_ts_scatter.draw()
        canvas_ts_scatter.get_tk_widget().pack(fill="both", expand=True, pady=5)

    # ----- 3D TAB -----
    tab3d = ScrollableFrame(notebook)
    notebook.add(tab3d, text="3D Graphs")

    fig3d = create_3d_figure(X_numeric, y)
    if fig3d is not None:
        canvas3d = FigureCanvasTkAgg(fig3d, master=tab3d.scrollable_frame)
        canvas3d.draw()
        canvas3d.get_tk_widget().pack(fill="both", expand=True, pady=10)

    fig3d_time = create_3d_time_figure(df_clean, datetime_cols, numeric_cols_all)
    if fig3d_time is not None:
        canvas3d_time = FigureCanvasTkAgg(fig3d_time, master=tab3d.scrollable_frame)
        canvas3d_time.draw()
        canvas3d_time.get_tk_widget().pack(fill="both", expand=True, pady=10)

    # Proper close handler so closing the window ends the program
    def on_close():
        try:
            root.destroy()
        finally:
            sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)

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

    print("\nCSV loaded successfully.")
    print(f"Shape of data: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nFirst 5 rows:\n")
    print(df.head())

    print("\nColumns:\n")
    print(list(df.columns))

    # 2.5 Analyze columns BEFORE dropping rows, so we see full structure
    df_for_analysis = df.copy()
    numeric_cols_all, categorical_cols, datetime_cols = analyze_columns(df_for_analysis)

    # Build and print suggestions (also used in Possible Graphs tab)
    suggestions_text = build_plot_suggestions_text(
        numeric_cols_all, categorical_cols, datetime_cols
    )

    # 3. Drop rows with missing values (for ML & clean plotting)
    df = df.dropna()
    print(f"\nAfter dropping rows with missing values: {df.shape[0]} rows remain")

    if df.shape[0] == 0:
        print("No data left after removing missing values. Please clean your CSV.")
        raise SystemExit

    # 4. Auto-select target: LAST column
    target_col = df.columns[-1]
    print(f"\nAuto-selected TARGET column: '{target_col}'")

    y = df[target_col]
    X_full = df.drop(columns=[target_col])

    # 5. Keep only numeric feature columns for ML
    X_numeric = X_full.select_dtypes(include=["int64", "float64"])

    if X_numeric.empty:
        print("\nNo numeric feature columns found to train on.")
        print("Hint: Add some numeric columns (for example, counts, amounts, measurements).")
        df_clean = df_for_analysis
        open_dashboard(df_clean, X_numeric, y,
                       numeric_cols_all, categorical_cols, datetime_cols,
                       suggestions_text)
        return

    print("\nUsing these feature columns for ML:\n", list(X_numeric.columns))

    # 6. Decide classification vs regression (basic heuristic)
    n_unique = y.nunique()
    is_classification = (y.dtype == "O") or (n_unique <= 20)
    problem_type = "classification" if is_classification else "regression"
    print(f"\nProblem type detected: {problem_type} (unique target values: {n_unique})")

    # Guard: avoid huge multi-class classification for this beginner demo
    if is_classification and n_unique > 50:
        print(f"\nWarning: Target '{target_col}' has {n_unique} unique values.")
        print("Too many classes for this simple demo. Skipping model training, showing graphs only.")
        df_clean = df_for_analysis
        open_dashboard(df_clean, X_numeric, y,
                       numeric_cols_all, categorical_cols, datetime_cols,
                       suggestions_text)
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
        print("\nSample predictions vs actual (first 10):")
        print("Predictions:", list(y_pred[:10]))
        print("Actual     :", list(y_test[:10].values))
        print(f"\nAccuracy: {acc * 100:.2f}%")

    else:
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print("\nSample predictions vs actual (first 10):")
        print("Predictions:", [round(v, 2) for v in y_pred[:10]])
        print("Actual     :", list(y_test[:10].values))
        print(f"\nMean Squared Error: {mse:.2f}")

    # 9. Open the GUI dashboard with tabs for Possible Graphs, 2D, 3D
    df_clean = df_for_analysis  # includes parsed datetimes
    open_dashboard(df_clean, X_numeric, y,
                   numeric_cols_all, categorical_cols, datetime_cols,
                   suggestions_text)


if __name__ == "__main__":
    main()
