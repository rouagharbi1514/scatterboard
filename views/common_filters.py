# Add or update this function


def add_date_filter(layout, callback=None):
    """Add date range filter to layout."""
    from PySide6.QtWidgets import QLabel, QDateEdit, QHBoxLayout, QWidget
    from PySide6.QtCore import QDate
    import data
    import pandas as pd

    # Get data to determine min/max dates
    df = data.get_dataframe()

    # Default date range (1 year)
    default_end = QDate.currentDate()
    default_start = default_end.addDays(-365)

    # If data is available, use its date range
    if df is not None and "date" in df.columns:
        # Get min and max dates from dataframe
        min_date = pd.to_datetime(df["date"].min())
        max_date = pd.to_datetime(df["date"].max())

        # Convert to QDate objects
        start_date = QDate(min_date.year, min_date.month, min_date.day)
        end_date = QDate(max_date.year, max_date.month, max_date.day)
    else:
        # Use default dates if data not available
        start_date = default_start
        end_date = default_end

    date_widget = QWidget()
    date_layout = QHBoxLayout(date_widget)
    date_layout.setContentsMargins(0, 0, 0, 0)

    # Add Start Date
    start_label = QLabel("Start Date:")
    start_label.setStyleSheet("color: #aaaaaa; font-weight: bold;")
    date_layout.addWidget(start_label)

    start_date_edit = QDateEdit()
    start_date_edit.setDate(start_date)
    start_date_edit.setCalendarPopup(True)
    start_date_edit.setDisplayFormat("yyyy-MM-dd")
    date_layout.addWidget(start_date_edit)

    date_layout.addSpacing(20)

    # Add End Date
    end_label = QLabel("End Date:")
    end_label.setStyleSheet("color: #aaaaaa; font-weight: bold;")
    date_layout.addWidget(end_label)

    end_date_edit = QDateEdit()
    end_date_edit.setDate(end_date)
    end_date_edit.setCalendarPopup(True)
    end_date_edit.setDisplayFormat("yyyy-MM-dd")
    date_layout.addWidget(end_date_edit)

    # Add to layout
    layout.addWidget(date_widget)

    # Connect signals if callback provided
    if callback:
        start_date_edit.dateChanged.connect(callback)
        end_date_edit.dateChanged.connect(callback)

    return start_date_edit, end_date_edit
