from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableView
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from data import get_feedback_data
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
    """Display feedback analysis."""
    widget = QWidget()
    layout = QVBoxLayout()

    # Add header
    header = QLabel("Feedback Analysis")
    header.setStyleSheet("font-size: 18pt; font-weight: bold;")
    layout.addWidget(header)

    # Get feedback data
    feedback_data = get_feedback_data()

    if feedback_data is not None:
        # Add description
        description = QLabel(
            "This view analyzes guest feedback to identify trends and improvement areas. "
            "Understanding guest sentiment helps enhance service quality."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Create table view
        table_view = QTableView()
        # Use PandasModel instead of direct conversion
        model = PandasModel(feedback_data)
        table_view.setModel(model)

        # Set table properties
        table_view.horizontalHeader().setStretchLastSection(True)
        table_view.setAlternatingRowColors(True)

        layout.addWidget(table_view)

        # Add sample analysis
        sample_analysis = QLabel(
            """
        <b>Feedback Summary:</b>
        <ul>
            <li>Overall satisfaction rating: 4.3/5.0</li>
            <li>Top positive mentions: Staff service, Room cleanliness</li>
            <li>Top improvement areas: Check-in process, Wi-Fi speed</li>
            <li>Net Promoter Score: 42 (Good)</li>
        </ul>
        """
        )
        layout.addWidget(sample_analysis)
    else:
        # Show message if no feedback data available
        no_data_label = QLabel(
            "Feedback analysis requires data with guest_satisfaction or feedback columns.\n"
            "Please upload data with this information to view the analysis."
        )
        no_data_label.setStyleSheet("font-size: 14pt; color: #ff9800;")
        no_data_label.setWordWrap(True)
        layout.addWidget(no_data_label)

    layout.addStretch()
    widget.setLayout(layout)

    return widget
