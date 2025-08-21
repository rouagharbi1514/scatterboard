# flake8: noqa
# views/utils.py – Modern UI helpers (Soft-UI), safe date ops & Plotly embed

import functools
import pandas as pd
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt, QSize
from data import is_data_loaded

# ─────────────────────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────────────────────
def data_required(view_func):
    """Ensure data is loaded before displaying a view. Shows a friendly card otherwise."""
    @functools.wraps(view_func)
    def wrapper(*args, **kwargs):
        if is_data_loaded():
            return view_func(*args, **kwargs)
        return create_error_widget("Please upload data before accessing this view")
    return wrapper


# ─────────────────────────────────────────────────────────────
# Error / Empty state (Soft-UI card)
# ─────────────────────────────────────────────────────────────
def _soft_card(content: QWidget, title: str | None = None) -> QWidget:
    """Wrap any widget in a soft card for consistent visual design."""
    card = QFrame()
    card.setObjectName("softCard")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(16, 16, 16, 16)
    if title:
        h = QLabel(title)
        h.setStyleSheet("color:#0f172a;font-size:13pt;font-weight:600;margin-bottom:8px;")
        lay.addWidget(h)
    lay.addWidget(content)
    card.setStyleSheet("""
        QFrame#softCard {
            background: #ffffff;
            border: 1px solid #e5e9f2;
            border-radius: 14px;
        }
    """)
    return card


def create_error_widget(message: str, details: str | None = None) -> QWidget:
    """Create a modern, friendly error/empty-state widget."""
    root = QWidget()
    root_lay = QVBoxLayout(root)
    root_lay.setContentsMargins(24, 24, 24, 24)
    root_lay.setAlignment(Qt.AlignCenter)

    card = QWidget()
    card_lay = QVBoxLayout(card)
    card_lay.setContentsMargins(24, 24, 24, 24)
    card.setStyleSheet("""
        QWidget {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #ffffff, stop:1 #fafbff);
            border: 1px solid #e5e9f2;
            border-radius: 18px;
        }
    """)

    # Icon + title
    icon = QLabel("⚠️")
    icon.setAlignment(Qt.AlignCenter)
    icon.setStyleSheet("font-size: 40pt; margin-bottom: 8px;")
    card_lay.addWidget(icon, 0, Qt.AlignCenter)

    title = QLabel(message)
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet("color:#0f172a;font-size:14pt;font-weight:700;")
    card_lay.addWidget(title, 0, Qt.AlignCenter)

    if details:
        sub = QLabel(details)
        sub.setAlignment(Qt.AlignCenter)
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#64748b;font-size:11pt;margin-top:6px;")
        card_lay.addWidget(sub, 0, Qt.AlignCenter)

    hint = QLabel("Try uploading your dataset (CSV/Excel) or use demo data to explore the app.")
    hint.setAlignment(Qt.AlignCenter)
    hint.setWordWrap(True)
    hint.setStyleSheet("color:#64748b;font-size:11pt;margin-top:12px;")
    card_lay.addWidget(hint, 0, Qt.AlignCenter)

    # CTA row
    cta_row = QHBoxLayout()
    cta_row.setAlignment(Qt.AlignCenter)

    demo_btn = QPushButton("Load Demo Data")
    demo_btn.setCursor(Qt.PointingHandCursor)
    demo_btn.setStyleSheet("""
        QPushButton {
            background-color: #7aa2f7;
            color: white;
            padding: 10px 18px;
            border: 0;
            border-radius: 10px;
            font-size: 11.5pt;
            font-weight: 600;
        }
        QPushButton:hover { background-color: #6b8df0; }
    """)
    demo_btn.clicked.connect(lambda: load_demo_data())
    cta_row.addWidget(demo_btn, 0, Qt.AlignCenter)

    card_lay.addLayout(cta_row)
    root_lay.addWidget(card, 0, Qt.AlignCenter)
    return root


def load_demo_data():
    """Ask the main window (if present) to load demo data."""
    from PySide6.QtWidgets import QApplication
    main_window = None
    for w in QApplication.topLevelWidgets():
        if w.objectName() == "MainWindow":
            main_window = w
            break
    if main_window and hasattr(main_window, "load_demo_data"):
        main_window.load_demo_data()


# ─────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────
def safe_date_diff(end_date, start_date) -> int:
    """Safely calculate (end - start) in days, tolerant to types."""
    try:
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date)
        return (end_date - start_date).days
    except Exception as e:
        print(f"[safe_date_diff] {e}")
        return 0


def filter_by_date(df: pd.DataFrame, start_date, end_date, date_col: str = "date") -> pd.DataFrame:
    """Filter DataFrame by inclusive date range (auto-convert to datetime)."""
    try:
        if date_col not in df.columns:
            return df
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
        return df.loc[mask]
    except Exception as e:
        print(f"[filter_by_date] {e}")
        return df


def safe_filter_by_date(df: pd.DataFrame, start, end, date_col: str = "Date") -> pd.DataFrame:
    """Safely filter with guard rails; returns original df on any issue."""
    if df is None or df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        try:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        except BaseException:
            return df
    mask = (out[date_col] >= pd.Timestamp(start)) & (out[date_col] <= pd.Timestamp(end))
    return out.loc[mask]


# ─────────────────────────────────────────────────────────────
# Plotly embedding (modern, no temp files)
# ─────────────────────────────────────────────────────────────
def create_plotly_widget(fig) -> QWidget:
    """
    Return a QWidget with an embedded Plotly figure (fast, no temp files).
    """
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
    from PySide6.QtWidgets import QSizePolicy as QSP, QWidget  # <-- import QWidget + QSizePolicy

    html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    view = QWebEngineView()
    view.setHtml(html)
    view.settings().setAttribute(
        QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
    )

    # ---- FIX: utiliser l'énum, pas l'instance ----
    Policy = getattr(QSP, "Policy", QSP)  # PySide6 récent: QSP.Policy ; ancien: QSP
    view.setSizePolicy(Policy.Expanding, Policy.Expanding)
    view.setMinimumHeight(450)  # optionnel, pour un rendu confortable
    # ---------------------------------------------

    return view


# ─────────────────────────────────────────────────────────────
# KPI Tile (Soft-UI, couleurs douces)
# ─────────────────────────────────────────────────────────────
def kpi_tile(label: str, value: str) -> QWidget:
    """
    Aesthetic KPI tile with soft gradient, subtle border and strong typography.
    API kept identical: kpi_tile(label, value) -> QWidget
    """
    tile = QWidget()
    tile.setMinimumSize(QSize(180, 96))
    tile.setStyleSheet("""
        QWidget {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #ffffff, stop:1 #fafbff);
            border: 1px solid #e5e9f2;
            border-radius: 14px;
        }
    """)

    lay = QVBoxLayout(tile)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(6)

    # Value (grand)
    v = QLabel(value)
    v.setAlignment(Qt.AlignCenter)
    v.setStyleSheet("font-size:22pt;font-weight:800;color:#0f172a;")
    lay.addWidget(v)

    # Label (petit, doux)
    l = QLabel(label)
    l.setAlignment(Qt.AlignCenter)
    l.setStyleSheet("font-size:10pt;font-weight:600;color:#64748b;")
    lay.addWidget(l)

    return tile


# ─────────────────────────────────────────────────────────────
# Data access shim (kept for backward-compat)
# ─────────────────────────────────────────────────────────────
def get_df():
    """Return the currently loaded dataframe from the data module."""
    from data import get_dataframe
    return get_dataframe()


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────
def format_currency(value, with_sign: bool = False) -> str:
    """
    Format a number as SAR currency with compact suffixes.
    Keeps original semantics (K/M) but with nicer spacing.
    """
    try:
        val = float(value)
    except Exception:
        return str(value)

    prefix = "+" if with_sign and val > 0 else ""
    a = abs(val)
    if a >= 1_000_000:
        return f"{prefix}{a/1_000_000:.1f}M SAR"
    if a >= 1_000:
        return f"{prefix}{a/1_000:.1f}K SAR"
    return f"{prefix}{a:.0f} SAR"
