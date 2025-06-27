import functools
import pandas as pd  # Add this import at the top
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt
from data import is_data_loaded


def data_required(view_func):
    """Decorator to ensure data is loaded before displaying a view."""

    @functools.wraps(view_func)
    def wrapper(*args, **kwargs):
        if is_data_loaded():
            # Data is available, proceed with the view
            return view_func(*args, **kwargs)
        else:
            # No data available, show message
            return create_error_widget("Please upload data before accessing this view")

    return wrapper


def create_error_widget(message, details=None):
    """Create an error widget with the given message."""
    widget = QWidget()
    layout = QVBoxLayout(widget)

    error_icon = QLabel("⚠️")
    error_icon.setStyleSheet("font-size: 48pt; color: #fbbf24;")
    layout.addWidget(error_icon, 0, Qt.AlignCenter) # type: ignore

    error_label = QLabel(message)
    error_label.setStyleSheet("color: #ff6b6b; font-size: 14pt;")
    layout.addWidget(error_label, 0, Qt.AlignCenter)

    if details:
        details_label = QLabel(details)
        details_label.setStyleSheet("color: #aaaaaa; font-size: 11pt;")
        layout.addWidget(details_label, 0, Qt.AlignCenter)

    help_text = QLabel(
        "Try uploading data with the required columns or using the demo data."
    )
    help_text.setStyleSheet("color: #aaaaaa; font-size: 12pt;")
    layout.addWidget(help_text, 0, Qt.AlignCenter)

    # Add button to load demo data
    demo_btn = QPushButton("Load Demo Data")
    demo_btn.setStyleSheet(
        """
        QPushButton {
            background-color: #0d6efd;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 12pt;
            margin-top: 20px;
        }
        QPushButton:hover {
            background-color: #0056b3;
        }
    """
    )

    # Connect to load demo data function
    demo_btn.clicked.connect(lambda: load_demo_data())
    layout.addWidget(demo_btn, 0, Qt.AlignCenter)

    return widget


def load_demo_data():
    """Load demo data from the main window."""
    # Find main window instance
    from PySide6.QtWidgets import QApplication

    main_window = None
    for widget in QApplication.topLevelWidgets():
        if widget.objectName() == "MainWindow":
            main_window = widget
            break

    if main_window and hasattr(main_window, "load_demo_data"):
        main_window.load_demo_data()


# Add this helper function for safe date operations


def safe_date_diff(end_date, start_date):
    """Safely calculate difference between dates."""
    import pandas as pd

    try:
        # Convert to pandas Timestamps if needed
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date)

        # Calculate difference in days
        diff = (end_date - start_date).days
        return diff
    except Exception as e:
        print(f"Error calculating date difference: {e}")
        return 0  # Return 0 as fallback


def filter_by_date(df, start_date, end_date, date_col="date"):
    """Filter dataframe by date range."""
    try:
        import pandas as pd

        # Make sure dates are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        # Convert filter dates to pandas Timestamps
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # Filter the dataframe
        return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    except Exception as e:
        print(f"Error filtering dataframe by date: {e}")
        return df  # Return original dataframe as fallback


# Add a safer date filtering function


def safe_filter_by_date(df, start, end, date_col="Date"):
    """Safely filter a dataframe by date range, handling errors gracefully."""
    if df is None or df.empty:
        return df

    if date_col not in df.columns:
        return df

    # Ensure column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            # Try to convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
        except BaseException:
            # If conversion fails, return original dataframe
            return df

    mask = (df[date_col] >= start) & (df[date_col] <= end)
    return df[mask]


def create_canvas(figure):
    """Create a canvas widget for a matplotlib figure."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    canvas = FigureCanvasQTAgg(figure)
    return canvas


def kpi_tile(label: str, value: str) -> QWidget:
    """
    Create a nice looking KPI tile with a label and value.

    Args:
        label: The name of the KPI
        value: The formatted value to display

    Returns:
        A QWidget containing the styled KPI tile
    """
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
    from PySide6.QtCore import Qt, QSize

    tile = QWidget()
    tile.setMinimumSize(QSize(180, 100))
    tile.setStyleSheet(
        """
        QWidget {
            background-color: #2d3748;
            border-radius: 8px;
            border: 1px solid #4a5568;
        }
    """
    )

    layout = QVBoxLayout(tile)
    layout.setContentsMargins(10, 10, 10, 10)

    # Value
    value_label = QLabel(value)
    value_label.setStyleSheet(
        """
        font-size: 20pt;
        font-weight: bold;
        color: #63b3ed;
    """
    )
    value_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(value_label)

    # Label
    name_label = QLabel(label)
    name_label.setStyleSheet(
        """
        font-size: 10pt;
        color: #e2e8f0;
    """
    )
    name_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(name_label)

    return tile


def get_df():
    """
    Returns the currently loaded dataframe from the data module.
    """
    from data import get_dataframe
    return get_dataframe()


def create_plotly_widget(fig):
    """
    Creates a QWidget containing a Plotly figure
    """
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtCore import QUrl
    import plotly
    import tempfile
    import os

    # Create a temporary HTML file
    temp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
    plotly.offline.plot(fig, filename=temp_file.name, auto_open=False)

    # Create web view widget
    web_view = QWebEngineView()
    web_view.load(QUrl.fromLocalFile(os.path.abspath(temp_file.name)))

    return web_view

def format_currency(value, with_sign=False):
    """
    Format a number as currency in SAR
    """
    prefix = ""
    if with_sign and value > 0:
        prefix = "+"

    if abs(value) >= 1000000:
        return f"{prefix}{abs(value)/1000000:.1f}M SAR"
    elif abs(value) >= 1000:
        return f"{prefix}{abs(value)/1000:.1f}K SAR"
    else:
        return f"{prefix}{abs(value):.0f} SAR"
