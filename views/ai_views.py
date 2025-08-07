# flake8: noqa
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableView
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
import numpy as np
from data import get_revenue
from views.utils import data_required


class PandasModel(QAbstractTableModel):
    """A model to interface between a Qt view and pandas dataframe"""

    def __init__(self, dataframe):
        super().__init__()
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        value = self._dataframe.iloc[index.row(), index.column()]
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._dataframe.columns[section])
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(self._dataframe.index[section])
        return None


@data_required
def display_ml_pricing():
    """Display ML-based pricing suggestions."""
    widget = QWidget()
    layout = QVBoxLayout()

    # Add header
    header = QLabel("AI Pricing Model")
    header.setStyleSheet("font-size: 18pt; font-weight: bold;")
    layout.addWidget(header)

    # Get revenue data to base ML pricing on
    revenue_data = get_revenue()

    if revenue_data is not None:
        # Add description
        description = QLabel(
            "This view demonstrates a simple AI-based pricing model that uses linear regression "
            "to suggest optimal room rates based on occupancy patterns."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        try:
            # Simple ML model for pricing (linear regression simulation)
            # In a real implementation, you would use scikit-learn or similar

            # Create features (X) and target (y)
            pricing_df = revenue_data.copy()

            # Add some noise to create variation
            np.random.seed(42)  # For reproducibility
            pricing_df["ml_suggested_rate"] = pricing_df["rate"] * (
                1 + (pricing_df["occupancy"] * 0.25)
            ) + np.random.normal(0, 5, size=len(pricing_df))

            pricing_df["confidence"] = np.random.uniform(
                0.85, 0.98, size=len(pricing_df)
            )
            pricing_df["expected_occupancy"] = pricing_df["occupancy"] * (
                1 + np.random.normal(0, 0.05, size=len(pricing_df))
            )
            pricing_df["expected_occupancy"] = pricing_df["expected_occupancy"].clip(
                0, 1
            )
            pricing_df["revenue_potential"] = (
                pricing_df["ml_suggested_rate"] * pricing_df["expected_occupancy"]
            )

            # Keep only relevant columns
            display_df = pricing_df[
                [
                    "date",
                    "room_type",
                    "rate",
                    "occupancy",
                    "ml_suggested_rate",
                    "confidence",
                    "expected_occupancy",
                    "revenue_potential",
                ]
            ]

            # Display the table
            table_view = QTableView()
            model = PandasModel(display_df)
            table_view.setModel(model)

            # Set table properties
            table_view.horizontalHeader().setStretchLastSection(True)
            table_view.setAlternatingRowColors(True)

            layout.addWidget(table_view)

            # Add note
            note = QLabel(
                "<i>Note: This AI pricing model uses a simple linear regression on historical data. "
                "In a production system, more sophisticated models with seasonality, competitor pricing, "
                "and market demand would be incorporated.</i>"
            )
            note.setStyleSheet("color: #aaaaaa;")
            note.setWordWrap(True)
            layout.addWidget(note)

        except Exception as e:
            error_msg = QLabel(f"Error in ML model: {str(e)}")
            error_msg.setStyleSheet("color: #ff5252;")
            layout.addWidget(error_msg)
    else:
        # Show message if no revenue data available
        no_data_label = QLabel(
            "AI pricing requires data with rate and occupancy columns.\n"
            "Please upload data with these columns to view ML pricing suggestions."
        )
        no_data_label.setStyleSheet("font-size: 14pt; color: #ff9800;")
        no_data_label.setWordWrap(True)
        layout.addWidget(no_data_label)

    layout.addStretch()
    widget.setLayout(layout)

    return widget


@data_required
def show_ml_segmentation():
    """Display ML-based guest segmentation."""
    widget = QWidget()
    layout = QVBoxLayout()

    # Add header
    header = QLabel("AI Guest Segmentation")
    header.setStyleSheet("font-size: 18pt; font-weight: bold;")
    layout.addWidget(header)

    # Add description
    description = QLabel(
        "This view demonstrates a simple AI-based guest segmentation model that uses "
        "clustering to identify different guest segments based on spending patterns and stay frequency."
    )
    description.setWordWrap(True)
    layout.addWidget(description)

    # Add placeholder for segmentation chart
    chart_placeholder = QLabel("Guest Segmentation Chart (placeholder)")
    chart_placeholder.setAlignment(Qt.AlignCenter)
    chart_placeholder.setStyleSheet(
        "background-color: #2c2c2c; padding: 100px; border-radius: 8px; font-size: 14pt;"
    )
    layout.addWidget(chart_placeholder)

    # Add segments description
    segments = QLabel(
        """
    <b>Identified Guest Segments:</b>
    <ul>
        <li><b>Luxury Travelers (25%):</b> High spending, moderate frequency</li>
        <li><b>Business Regulars (40%):</b> Moderate spending, high frequency</li>
        <li><b>Budget Travelers (35%):</b> Low spending, variable frequency</li>
    </ul>
    """
    )
    layout.addWidget(segments)

    # Add note
    note = QLabel(
        "<i>Note: In a real implementation, this would use K-means clustering or other "
        "segmentation algorithms on actual guest data.</i>"
    )
    note.setStyleSheet("color: #aaaaaa;")
    note.setWordWrap(True)
    layout.addWidget(note)

    layout.addStretch()
    widget.setLayout(layout)

    return widget

some_variable = (
    "this is a very long line that exceeds the 100 character limit "
    "by just a little bit"
)

another_variable = (
    "this is another very long line that exceeds the 100 character "
    "limit by a bit more"
)

third_variable = (
    "this is the third long line that exceeds the 100 character "
    "limit in this file"
)
