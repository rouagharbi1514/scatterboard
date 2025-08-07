# flake8: noqa
# views/advanced.py – adds K-Means guest segmentation
"""
Advanced Analytics & Guest Segmentation
======================================
This module now contains two tabs:
1. **Overview** – your existing statistics table
2. **Guest Segmentation** – runs a K‑Means model (k=4 by default) on RFM‑style
   features and visualises clusters in a 2‑D scatter (PCA) plus a summary table.

Dependencies: scikit‑learn ≥ 1.0 (already lightweight). If packaging with PyInstaller
just ensure `sklearn` is in requirements.txt.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableView, QTabWidget,
    QComboBox, QPushButton
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from views.utils import data_required  # kpi_tile import removed as unused
from data.helpers import get_dataframe


# ─────────────────────────────────────────────────────────────
# Qt <-> pandas model helper
# ─────────────────────────────────────────────────────────────
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__(); self._df = df
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)
    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)
    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid() or role != Qt.DisplayRole: return None
        return str(self._df.iat[idx.row(), idx.column()])
    def headerData(self, sec, orient, role=Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return str(self._df.columns[sec]) if orient==Qt.Horizontal else str(self._df.index[sec])


# ─────────────────────────────────────────────────────────────
# Feature engineering for segmentation
# ─────────────────────────────────────────────────────────────

def _build_guest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency‑Frequency‑Monetary style features per GuestID."""
    if "GuestID" not in df.columns or "date" not in df.columns or "revenue" not in df.columns:
        raise ValueError("Dataset lacks GuestID, date, or revenue columns.")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    snapshot = work["date"].max() + pd.Timedelta(days=1)
    grouped = work.groupby("GuestID").agg(
        recency=("date", lambda x: (snapshot - x.max()).days),
        frequency=("date", "count"),
        monetary=("revenue", "sum"),
        avg_stay=("LOS", "mean") if "LOS" in work.columns else ("date", "count")
    )
    grouped = grouped.fillna(0)
    return grouped


# ─────────────────────────────────────────────────────────────
# Main Qt view
# ─────────────────────────────────────────────────────────────
@data_required
def display() -> QWidget:  # noqa: C901
    df = get_dataframe()
    root = QWidget(); root.setLayout(QVBoxLayout())
    header = QLabel("Advanced Analytics & Segmentation"); header.setStyleSheet("font-size:18pt;font-weight:bold;")
    root.layout().addWidget(header)

    if df is None or df.empty:
        root.layout().addWidget(QLabel("Upload data first.")); return root

    tabs = QTabWidget(); root.layout().addWidget(tabs, 1)

    # ── Overview tab (existing content trimmed) ──────────────────────────
    overview = QWidget(); overview.setLayout(QVBoxLayout())
    overview.layout().addWidget(QLabel("<b>Quick Stats</b>"))
    rev_table = QTableView(); model = PandasModel(df.head(15))
    rev_table.setModel(model); rev_table.horizontalHeader().setStretchLastSection(True)
    overview.layout().addWidget(rev_table)
    tabs.addTab(overview, "Overview")

    # ── Segmentation tab ────────────────────────────────────────────────
    seg_tab = QWidget(); seg_tab.setLayout(QVBoxLayout())
    info_lbl = QLabel("Choose <i>k</i> and click <b>Run Clustering</b>.")
    seg_tab.layout().addWidget(info_lbl)

    top_row = QHBoxLayout(); seg_tab.layout().addLayout(top_row)
    top_row.addWidget(QLabel("#Clusters:"))
    k_combo = QComboBox(); k_combo.addItems(["3","4","5","6"]); top_row.addWidget(k_combo)
    run_btn = QPushButton("Run Clustering"); top_row.addWidget(run_btn); top_row.addStretch()

    # placeholder widgets (figure + table)
    fig_canvas = Canvas(Figure(figsize=(6,4))); seg_tab.layout().addWidget(fig_canvas)
    table = QTableView(); seg_tab.layout().addWidget(table)
    seg_tab.layout().addStretch()
    tabs.addTab(seg_tab, "Guest Segmentation")

    # ── clustering action ───────────────────────────────────────────────
    def _cluster():
        k = int(k_combo.currentText())
        feats = _build_guest_features(df)
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(feats)
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        feats["cluster"] = km.labels_

        # PCA to 2‑D for plotting ----------------------------------------
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        feats["pc1"], feats["pc2"] = coords[:,0], coords[:,1]

        # matplotlib scatter
        fig = fig_canvas.figure; fig.clear(); ax = fig.add_subplot(111)
        for cl in range(k):
            subset = feats[feats["cluster"]==cl]
            ax.scatter(subset["pc1"], subset["pc2"], label=f"Cluster {cl}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.setTitle("K‑Means Segments (PCA)"); ax.legend()
        fig.tight_layout(); fig_canvas.draw()

        # summary table ---------------------------------------------------
        summary = feats.groupby("cluster").agg(
            Guests = ("recency", "count"),
            AvgRecency = ("recency", "mean"),
            AvgFreq = ("frequency", "mean"),
            AvgMonetary = ("monetary", "mean")
        ).round(2)
        table.setModel(PandasModel(summary.reset_index()))
        table.horizontalHeader().setStretchLastSection(True)

    run_btn.clicked.connect(_cluster)
    return root


# ─────────────────────────────────────────────────────────────
# Dig Deeper Analytics - Enhanced visualization and insights
# ─────────────────────────────────────────────────────────────
@data_required
def display_dig_deeper() -> QWidget:
    """Display advanced analytics with deeper insights."""
    df = get_dataframe()
    root = QWidget()
    root.setLayout(QVBoxLayout())
    
    header = QLabel("Advanced Insights & Predictive Analytics")
    header.setStyleSheet("font-size: 18pt; font-weight: bold; color: white;")
    root.layout().addWidget(header)

    if df is None or df.empty:
        root.layout().addWidget(QLabel("Upload data first."))
        return root

    # Create tabs for different advanced analytics
    tabs = QTabWidget()
    root.layout().addWidget(tabs, 1)

    # ── Trend Analysis Tab ──────────────────────────────────────
    trend_tab = QWidget()
    trend_tab.setLayout(QVBoxLayout())
    
    # Convert date column and ensure it's sorted
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df_sorted = df.sort_values('date')
        
        # Create time series figure
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        # Plot key metrics over time
        if 'revpar' in df.columns:
            ax.plot(df_sorted['date'], df_sorted['revpar'], 'b-', label='RevPAR', linewidth=2)
        
        if 'goppar' in df.columns:
            ax.plot(df_sorted['date'], df_sorted['goppar'], 'g-', label='GOPPAR', linewidth=2)
            
        if 'rate' in df.columns:
            ax.plot(df_sorted['date'], df_sorted['rate'], 'r-', label='ADR', linewidth=2)
        
        ax.set_title("Key Metrics Trend Analysis", color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Value ($)", color='white')
        ax.tick_params(axis='both', colors='white')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas = Canvas(fig)
        trend_tab.layout().addWidget(canvas)
    else:
        trend_tab.layout().addWidget(QLabel("Date column not found in dataset."))
    
    tabs.addTab(trend_tab, "Trend Analysis")

    # ── Correlation Analysis Tab ──────────────────────────────────
    corr_tab = QWidget()
    corr_tab.setLayout(QVBoxLayout())
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty and numeric_df.shape[1] >= 2:
        # Create correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot heatmap
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        cmap = matplotlib.cm.coolwarm
        im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.index)):
                value = corr_matrix.iloc[j, i]
                color = 'white' if abs(value) < 0.7 else 'black'
                ax.text(i, j, f"{value:.2f}", ha='center', va='center', color=color)
        
        # Set ticks
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', color='white')
        ax.set_yticklabels(corr_matrix.index, color='white')
        
        ax.set_title("Correlation Matrix", color='white', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(colors='white')
        
        fig.tight_layout()
        canvas = Canvas(fig)
        corr_tab.layout().addWidget(canvas)
    else:
        corr_tab.layout().addWidget(QLabel("Insufficient numeric data for correlation analysis."))
    
    tabs.addTab(corr_tab, "Correlation Analysis")

    # ── Predictive Analytics Tab ──────────────────────────────────
    predict_tab = QWidget()
    predict_tab.setLayout(QVBoxLayout())
    
    predict_controls = QHBoxLayout()
    predict_tab.layout().addLayout(predict_controls)
    
    metric_label = QLabel("Target Metric:")
    predict_controls.addWidget(metric_label)
    
    # Dropdown for metric selection
    metric_combo = QComboBox()
    if not numeric_df.empty:
        metric_combo.addItems(numeric_df.columns)
    predict_controls.addWidget(metric_combo)
    
    # Run prediction button
    predict_btn = QPushButton("Run Prediction")
    predict_controls.addWidget(predict_btn)
    predict_controls.addStretch()
    
    # Placeholder for prediction chart
    chart_frame = QWidget()
    chart_layout = QVBoxLayout(chart_frame)
    chart_fig = Figure(figsize=(10, 6))
    chart_canvas = Canvas(chart_fig)
    chart_layout.addWidget(chart_canvas)
    predict_tab.layout().addWidget(chart_frame)
    
    # Function to run prediction
    def run_prediction():
        if 'date' not in df.columns or metric_combo.currentText() not in df.columns:
            return
        
        target = metric_combo.currentText()
        
        # Clear previous plot
        chart_fig.clear()
        ax = chart_fig.add_subplot(111)
        chart_fig.patch.set_facecolor('#2d2d2d')
        ax.set_facecolor('#2d2d2d')
        
        # Sort by date
        df_sorted = df.sort_values('date')
        
        # Simple moving average for prediction (as a placeholder for more complex models)
        window = min(7, len(df_sorted) // 3)
        df_sorted[f'{target}_ma'] = df_sorted[target].rolling(window=window, min_periods=1).mean()
        
        # Plot actual vs predicted
        ax.plot(df_sorted['date'], df_sorted[target], 'b-', label='Actual', alpha=0.7)
        ax.plot(df_sorted['date'], df_sorted[f'{target}_ma'], 'r-', label='Predicted (MA)', linewidth=2)
        
        # Add future prediction (simple extension of moving average)
        last_date = df_sorted['date'].max()
        future_dates = pd.date_range(start=last_date, periods=10)
        future_pred = [df_sorted[f'{target}_ma'].iloc[-1]] * 10
        ax.plot(future_dates, future_pred, 'r--', linewidth=2)
        
        ax.set_title(f"Prediction for {target}", color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel(target, color='white')
        ax.tick_params(axis='both', colors='white')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        chart_fig.tight_layout()
        chart_canvas.draw()
    
    predict_btn.clicked.connect(run_prediction)
    
    tabs.addTab(predict_tab, "Predictive Analytics")
    
    return root
