from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QComboBox,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, QThread, Signal
import os
from data import load_data


class DataLoaderThread(QThread):
    """Thread for loading data without freezing the UI."""

    progress_signal = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, file_path, file_type):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type

    def run(self):
        try:
            # Simulate progress updates
            for i in range(1, 101):
                self.progress_signal.emit(i)
                self.msleep(20)  # Sleep for 20ms

            # Actually load the data
            success = load_data(self.file_path)

            if success:
                self.finished_signal.emit(True, "Data loaded successfully!")
            else:
                self.finished_signal.emit(
                    False, "Failed to load data. Check the file format."
                )
        except Exception as e:
            self.finished_signal.emit(False, f"Error loading data: {str(e)}")


class FileUploadWidget(QWidget):
    """Widget for uploading data files."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Add header
        header = QLabel("Upload Data")
        header.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 10px;
        """
        )
        layout.addWidget(header)

        # Add description
        description = QLabel(
            "Upload your hotel data file to analyze KPIs, trends, and insights. "
            "The system supports CSV, Excel, and other common data formats."
        )
        description.setWordWrap(True)
        description.setStyleSheet(
            "font-size: 14px; color: #aaaaaa; margin-bottom: 15px;"
        )
        layout.addWidget(description)

        # File type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("File Type:")
        type_label.setStyleSheet("font-size: 14px;")
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["Auto-detect", "CSV", "Excel", "JSON"])
        self.file_type_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #2c2c2c;
                color: white;
                padding: 5px;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #2c2c2c;
                color: white;
                selection-background-color: #0d6efd;
            }
        """
        )
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.file_type_combo)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        # Upload button
        self.upload_button = QPushButton("Select File")
        self.upload_button.clicked.connect(self.select_file)
        self.upload_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """
        )
        layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                background-color: #2c2c2c;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
            }
        """
        )
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 14px; margin-top: 10px;")
        layout.addWidget(self.status_label)

        # File requirements
        requirements = QLabel(
            """
        <b>Data Requirements:</b>
        <ul>
            <li>The file should contain columns for dates, rates, and occupancy</li>
            <li>Optional columns: cost_per_occupied_room, guest_satisfaction, room_type</li>
            <li>Data should be clean and properly formatted</li>
            <li>Date column should be in a standard date format</li>
        </ul>
        """
        )
        requirements.setStyleSheet("font-size: 14px; color: #aaaaaa; margin-top: 20px;")
        layout.addWidget(requirements)

        # Add stretcher
        layout.addStretch()

    def select_file(self):
        """Open file dialog to select a data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "Data Files (*.csv *.xlsx *.xls *.json);;All Files (*)",
        )

        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path):
        """Load the selected file."""
        # Update UI
        self.upload_button.setEnabled(False)
        self.progress_bar.show()
        self.status_label.setText(f"Loading file: {os.path.basename(file_path)}")
        self.status_label.setStyleSheet(
            "font-size: 14px; color: #ffffff; margin-top: 10px;"
        )

        # Get selected file type
        file_type = self.file_type_combo.currentText()

        # Create and start loader thread
        self.loader_thread = DataLoaderThread(file_path, file_type)
        self.loader_thread.progress_signal.connect(self.update_progress)
        self.loader_thread.finished_signal.connect(self.loading_finished)
        self.loader_thread.start()

    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def loading_finished(self, success, message):
        """Handle the completion of data loading."""
        if success:
            self.status_label.setText(message)
            self.status_label.setStyleSheet(
                "font-size: 14px; color: #4CAF50; margin-top: 10px;"
            )
        else:
            self.status_label.setText(message)
            self.status_label.setStyleSheet(
                "font-size: 14px; color: #F44336; margin-top: 10px;"
            )

        # Reset UI
        self.upload_button.setEnabled(True)


def display():
    """Display file upload interface."""
    return FileUploadWidget()
