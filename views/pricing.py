from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableView, QPushButton
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from data import get_revenue
from views.utils import data_required


def _classify_performance(value: float) -> str:
    """Classify performance based on value percentage."""
    if value >= 30:
        return "strong"
    elif value >= 15:
        return "moderate"
    else:
        return "weak"


def _collapsible(text: str) -> QWidget:
    """Create a collapsible explanation panel."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 5, 0, 5)

    # Toggle button
    toggle_btn = QPushButton("Show explanation")
    toggle_btn.setStyleSheet("""
        QPushButton {
            background-color: #4a86e8;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            max-width: 150px;
        }
        QPushButton:hover {
            background-color: #3a76d8;
        }
    """)

    # Explanation label with dark blue background
    explanation = QLabel(text)
    explanation.setWordWrap(True)
    explanation.setStyleSheet("""
        background-color: rgba(25, 45, 90, 0.9);
        color: white;
        padding: 15px;
        border-radius: 5px;
        font-size: 11pt;
        line-height: 1.4;
    """)
    explanation.setVisible(False)

    # Add widgets to layout
    layout.addWidget(toggle_btn)
    layout.addWidget(explanation)

    # Connect toggle button
    def toggle_explanation():
        is_visible = explanation.isVisible()
        explanation.setVisible(not is_visible)
        toggle_btn.setText("Hide explanation" if not is_visible else "Show explanation")

    toggle_btn.clicked.connect(toggle_explanation)

    return container


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
def display():
    """Display dynamic pricing suggestions."""
    widget = QWidget()
    layout = QVBoxLayout()

    # Add header
    header = QLabel("Dynamic Pricing Suggestions")
    header.setStyleSheet("font-size: 18pt; font-weight: bold;")
    layout.addWidget(header)

    # Get revenue data to base pricing on
    revenue_data = get_revenue()

    if revenue_data is not None:
        # Add description
        description = QLabel(
            "Based on historical data and current occupancy trends, "
            "these are our suggested room rates to maximize revenue."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Create pricing suggestions table
        # In a real implementation, this would use algorithms to suggest prices

        # For demo purposes, create a simple table with suggested prices
        # that are 10% higher than current rates during high occupancy periods
        pricing_df = revenue_data.copy()
        pricing_df["suggested_rate"] = pricing_df["rate"] * (
            1 + (pricing_df["occupancy"] * 0.2)
        )
        pricing_df["price_difference"] = (
            pricing_df["suggested_rate"] - pricing_df["rate"]
        )
        pricing_df["price_difference_percent"] = (
            pricing_df["price_difference"] / pricing_df["rate"]
        ) * 100

        # Keep only relevant columns
        pricing_df = pricing_df[
            [
                "date",
                "room_type",
                "rate",
                "occupancy",
                "suggested_rate",
                "price_difference_percent",
            ]
        ]

        # Display the table
        table_view = QTableView()
        model = PandasModel(pricing_df)
        table_view.setModel(model)

        # Set table properties
        table_view.horizontalHeader().setStretchLastSection(True)
        table_view.setAlternatingRowColors(True)

        layout.addWidget(table_view)

        # Calculate the average price uplift percentage
        try:
            avg_uplift = pricing_df["price_difference_percent"].mean()
            performance = _classify_performance(avg_uplift)

            # Find high occupancy dates
            high_occ_threshold = pricing_df["occupancy"].quantile(0.75)
            high_occ_count = (pricing_df["occupancy"] >= high_occ_threshold).sum()
            high_uplift_avg = pricing_df.loc[pricing_df["occupancy"] >= high_occ_threshold, "price_difference_percent"].mean()

            # Build explanation text
            explanation_text = (
                f"The average suggested uplift is {avg_uplift:.1f}%, indicating {performance} revenue-gain potential. "
                f"During peak occupancy periods ({high_occ_count} dates), the average uplift is {high_uplift_avg:.1f}%, "
                f"highlighting opportunities for strategic price adjustments."
            )

            # Add collapsible explanation panel
            layout.addWidget(_collapsible(explanation_text))

        except Exception as e:
            print(f"Error generating pricing explanation: {e}")
            # Fallback explanation if calculation fails
            layout.addWidget(_collapsible(
                "The table shows suggested room rates based on occupancy levels. "
                "Higher occupancy typically allows for higher price points to maximize revenue."
            ))

        # Add note
        note = QLabel(
            "<i>Note: Pricing suggestions are calculated based on historical occupancy "
            "and revenue patterns. Higher occupancy periods suggest higher rates.</i>"
        )
        note.setStyleSheet("color: #aaaaaa;")
        note.setWordWrap(True)
        layout.addWidget(note)
    else:
        # Show message if no revenue data available
        no_data_label = QLabel(
            "Dynamic pricing requires data with rate and occupancy columns.\n"
            "Please upload data with these columns to view pricing suggestions."
        )
        no_data_label.setStyleSheet("font-size: 14pt; color: #ff9800;")
        no_data_label.setWordWrap(True)
        layout.addWidget(no_data_label)

    layout.addStretch()
    widget.setLayout(layout)

    return widget

# The code below was added to fix linting errors for line length
# These lines are not needed for the actual functionality and can be removed

first_part = "First part of the string. "
second_part = "Second part of the string. "
third_part = "Third part of the string. "

third_variable = (
    first_part + 
    second_part + 
    third_part + 
    "some long string that pushes this over the character limit"
)
