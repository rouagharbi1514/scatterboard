from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableView
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from data import get_retention_data
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
def display():
    """Display guest retention analysis."""
    widget = QWidget()
    layout = QVBoxLayout()

    # Add header
    header = QLabel("Guest Retention & Repeat Visits")
    header.setStyleSheet("font-size: 18pt; font-weight: bold;")
    layout.addWidget(header)

    # Get retention data
    retention_data = get_retention_data()

    if retention_data is not None:
        # Add description
        description = QLabel(
            "This view analyzes guest retention patterns and repeat visit behavior. "
            "Understanding these patterns helps optimize loyalty programs and marketing efforts."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Create table view
        table_view = QTableView()
        # Use PandasModel instead of DataFrame directly
        model = PandasModel(retention_data)
        table_view.setModel(model)

        # Set table properties
        table_view.horizontalHeader().setStretchLastSection(True)
        table_view.setAlternatingRowColors(True)

        layout.addWidget(table_view)

        # Add sample retention metrics
        retention_metrics = QLabel(
            """
        <b>Retention Metrics:</b>
        <ul>
            <li>Average guest return rate: 32%</li>
            <li>Average time between stays: 127 days</li>
            <li>Loyalty program participation: 45%</li>
            <li>Retention improvement YoY: +5.2%</li>
        </ul>
        """
        )
        layout.addWidget(retention_metrics)
    else:
        # Show message if no retention data available
        no_data_label = QLabel(
            "Guest retention analysis requires data with guest_id and visit frequency information.\n"
            "Please upload data with this information to view the analysis."
        )
        no_data_label.setStyleSheet("font-size: 14pt; color: #ff9800;")
        no_data_label.setWordWrap(True)
        layout.addWidget(no_data_label)

    layout.addStretch()
    widget.setLayout(layout)

    return widget
