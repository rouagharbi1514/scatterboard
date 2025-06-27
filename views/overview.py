from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas # type: ignore
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QPushButton,
    QFrame,
)
from PySide6.QtCore import Qt, QTimer, QAbstractTableModel
from data import get_kpis, get_dataframe
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")


class PandasTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data.columns)

    def data(self, index, role=Qt.DisplayRole): # type: ignore
        if role == Qt.DisplayRole: # type: ignore
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return str(self._data.columns[section])
        return None


class OverviewWidget(QWidget):
    """Widget displaying an overview of hotel metrics."""

    def __init__(self):
        super().__init__()
        self.setObjectName("overview_widget")
        self.data = None
        self.kpi_data = None
        self.init_ui()

        # Set up timer to refresh data every minute
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(60000)  # Refresh every minute

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Add header
        header = QLabel("Hotel Dashboard Overview")
        header.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 10px;
        """
        )
        main_layout.addWidget(header)

        # Add description
        description = QLabel(
            "Welcome to the Hotel Dashboard. This overview shows key metrics and trends "
            "from your hotel data. Use the navigation menu to explore detailed analyses."
        )
        description.setWordWrap(True)
        description.setStyleSheet(
            "font-size: 14px; color: #aaaaaa; margin-bottom: 15px;"
        )
        main_layout.addWidget(description)

        # Create grid for KPI cards and charts
        content_layout = QGridLayout()
        content_layout.setSpacing(15)
        main_layout.addLayout(content_layout)

        # Add KPI summary card
        self.kpi_card = QFrame()
        self.kpi_card.setStyleSheet(
            """
            background-color: #2c2c2c;
            border-radius: 8px;
            padding: 15px;
        """
        )
        kpi_layout = QVBoxLayout(self.kpi_card)

        kpi_header = QLabel("Key Performance Indicators")
        kpi_header.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        kpi_layout.addWidget(kpi_header)

        self.kpi_content = QLabel("Loading data...")
        self.kpi_content.setStyleSheet("font-size: 14px; color: #dddddd;")
        self.kpi_content.setTextFormat(Qt.RichText)
        kpi_layout.addWidget(self.kpi_content)

        content_layout.addWidget(self.kpi_card, 0, 0, 1, 1)

        # Add occupancy chart
        self.occupancy_chart = QFrame()
        self.occupancy_chart.setStyleSheet(
            """
            background-color: #2c2c2c;
            border-radius: 8px;
            padding: 15px;
        """
        )
        occupancy_layout = QVBoxLayout(self.occupancy_chart)

        occupancy_header = QLabel("Occupancy Trend")
        occupancy_header.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #ffffff;"
        )
        occupancy_layout.addWidget(occupancy_header)

        self.occupancy_figure = Figure(figsize=(5, 3), dpi=100)
        self.occupancy_canvas = Canvas(self.occupancy_figure)
        occupancy_layout.addWidget(self.occupancy_canvas)

        content_layout.addWidget(self.occupancy_chart, 0, 1, 1, 1)

        # Add refresh button
        refresh_button = QPushButton("Refresh Data")
        refresh_button.clicked.connect(self.refresh_data)
        refresh_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """
        )
        main_layout.addWidget(refresh_button, alignment=Qt.AlignRight)

        # Add stretcher
        main_layout.addStretch()

        # Initial data load
        self.refresh_data()

    def refresh_data(self):
        """Refresh the data displayed in the overview."""
        # Get the data
        self.data = get_dataframe()
        self.kpi_data = get_kpis()

        # Update KPI content
        if self.kpi_data and "message" not in self.kpi_data:
            # Format KPI values
            avg_occ = self.kpi_data.get("average_occupancy", 0)
            avg_rate = self.kpi_data.get("average_rate", 0)
            revpar = self.kpi_data.get("revpar", 0)
            goppar = self.kpi_data.get("goppar", 0)

            # Calculate total revenue if possible
            total_revenue = 0
            if (
                self.data is not None
                and "rate" in self.data.columns
                and "occupancy" in self.data.columns
            ):
                total_revenue = (self.data["rate"] * self.data["occupancy"]).sum()

            # Format the KPI text with HTML
            kpi_text = f"""
            <b>Average Occupancy:</b> {avg_occ:.1f}%<br>
            <b>Average Daily Rate:</b> ${avg_rate:.2f}<br>
            <b>RevPAR:</b> ${revpar:.2f}<br>
            <b>GOPPAR:</b> ${goppar:.2f}<br>
            <b>Total Revenue:</b> ${total_revenue:.2f}<br>
            """

            self.kpi_content.setText(kpi_text)
        else:
            error_msg = "No KPI data available"
            if self.kpi_data and "message" in self.kpi_data:
                error_msg = self.kpi_data["message"]
            self.kpi_content.setText(f"<i>{error_msg}</i>")

        # Update occupancy chart
        if (
            self.data is not None
            and "date" in self.data.columns
            and "occupancy" in self.data.columns
        ):
            try:
                # Ensure date is datetime
                if not pd.api.types.is_datetime64_any_dtype(self.data["date"]):
                    self.data["date"] = pd.to_datetime(self.data["date"])

                # Group by month and calculate average occupancy
                monthly_occupancy = self.data.groupby(
                    self.data["date"].dt.to_period("M")
                )["occupancy"].mean()

                # Clear the figure
                self.occupancy_figure.clear()

                # Create new plot
                ax = self.occupancy_figure.add_subplot(111)

                # Plot the data
                months = [period.to_timestamp() for period in monthly_occupancy.index]
                ax.plot(
                    months,
                    monthly_occupancy.values,
                    marker="o",
                    linestyle="-",
                    color="#4CAF50",
                )

                # Format the plot
                ax.set_ylim(0, 1)
                ax.set_xlabel("")
                ax.grid(True, linestyle="--", alpha=0.7)

                # Format y-axis as percentage
                vals = ax.get_yticks()
                ax.set_yticklabels([f"{x:.0%}" for x in vals])

                # Format x-axis to show month names
                import matplotlib.dates as mdates

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

                # Update the canvas
                self.occupancy_figure.tight_layout()
                self.occupancy_canvas.draw()
            except Exception as e:
                print(f"Error updating occupancy chart: {e}")

                # Show error on chart
                self.occupancy_figure.clear()
                ax = self.occupancy_figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    f"Error: {str(e)}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="#FF9800",
                )
                ax.axis("off")
                self.occupancy_figure.tight_layout()
                self.occupancy_canvas.draw()


def display():
    """Display hotel dashboard overview."""
    return OverviewWidget()
