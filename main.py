#!/usr/bin/env python3
"""
Hotel Dashboard – Main Application
===================================

Qt-based interactive analytics suite for hotel data.
The file is reformatted to satisfy **flake8** (PEP-8, line-length ≤ 79)
without altering behaviour.
"""

from __future__ import annotations
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from PySide6.QtCore import (
    Qt,
    QEasingCurve,
    QPropertyAnimation,
    QTimer,
    QDateTime,
)
from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from routes_grouped import ROUTES_GROUPED
import views                                 # ✨ NEW – needed for file-upload view
import data  # provides ``load_dataframe``
import resources_rc  # noqa: F401  (import registers Qt resources at runtime)
from auth import authenticate_user

# ─────────────────────────────── application window


class MainWindow(QMainWindow):
    """Primary window containing sidebar and stacked content views."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hotel Analytics Dashboard")
        self.setObjectName("MainWindow")

        # reasonable defaults
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        # central widget & layout (zero margins)
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # main container with inner shadow/padding
        main_container = QFrame(objectName="mainContainer")
        main_container_layout = QVBoxLayout(main_container)
        main_container_layout.setContentsMargins(10, 10, 10, 10)
        main_container_layout.setSpacing(0)
        vbox.addWidget(main_container)

        # ───────── header bar
        self.header = QFrame(objectName="header")
        self.header.setFixedHeight(50)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(15, 0, 15, 0)

        title = QLabel("ScatterBoard", objectName="headerTitle")
        header_layout.addWidget(title)

        # Add spacer to push time and version to the right
        header_layout.addStretch()

        # Add Saudi Arabia time display
        self.time_label = QLabel("", objectName="timeLabel")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.time_label)

        # Add some spacing between the time and version
        header_layout.addSpacing(10)

        version = QLabel("v3.0", objectName="versionLabel")
        version.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(version)

        main_container_layout.addWidget(self.header)

        # ───────── splitter → sidebar | content
        content_container = QFrame(objectName="contentContainer")
        content_layout = QHBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        main_container_layout.addWidget(content_container)

        self.split = QSplitter(Qt.Horizontal)
        content_layout.addWidget(self.split)

        self.sidebar = self._create_sidebar()

        self.content_frame = QFrame(objectName="contentFrame")
        content_frame_layout = QVBoxLayout(self.content_frame)
        content_frame_layout.setContentsMargins(20, 20, 20, 20)

        self.content = QStackedWidget()
        content_frame_layout.addWidget(self.content)

        self.split.addWidget(self.sidebar)
        self.split.addWidget(self.content_frame)
        self.split.setStretchFactor(1, 1)
        self.split.setHandleWidth(1)

        # demo & welcome screen
        self._load_demo_data()
        self._show_welcome()

        # global stylesheet
        # Initialize theme and apply stylesheet
        self.current_theme = "light"  # default theme
        self._apply_stylesheet()

        # Start timer to update the time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Update every second

    # ──────────────────────── stylesheet
    def _apply_stylesheet(self) -> None:
        """Apply the stylesheet for the current theme."""

        if getattr(self, "current_theme", "light") == "dark":
            css = self._stylesheet_dark()
            if hasattr(self, "theme_toggle_btn"):
                self.theme_toggle_btn.setText("☀️ Light Mode")
        else:
            css = self._stylesheet_light()
            if hasattr(self, "theme_toggle_btn"):
                self.theme_toggle_btn.setText("🌙 Dark Mode")
        self.setStyleSheet(css)

    # pulled out to keep line-length sane
    @staticmethod
    def _stylesheet_light() -> str:
        """Return light theme Qt CSS with a refined luxury aesthetic."""
        return (
            """
            /* App base */
            QMainWindow { background: #F7F8FB; color: #2A2E36; }
            #mainContainer { background:#F7F8FB; border-radius:12px; border:1px solid #E7EAF0; }

            /* Header: soft ivory + gold accent */
            #header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0E1F3C, stop:1 #19355F);
                border-bottom: 3px solid #D6B56A;
                border-top-left-radius:12px;
                border-top-right-radius:12px;
                color:#FFFFFF;
            }
            #headerTitle { font-family:'Montserrat','Segoe UI',Arial,sans-serif; font-size:20px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#FFFFFF; }
            #timeLabel { font-family:'Open Sans','Segoe UI',Arial; font-size:13px; color:#E9D9A4; font-weight:600; }
            #versionLabel { color:#C5CFDF; font-size:12px; }

            /* Content frame card */
            #contentFrame {
                background:#FFFFFF;
                border-radius:16px;
                margin:12px;
                border:1px solid #E7EAF0;
            }

            /* Sidebar with gold rail */
            #sidebar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0E1F3C, stop:1 #19355F);
                border-right: 3px solid #D6B56A;
                min-width:300px; max-width:300px; color:#FFFFFF;
            }
            #sidebar QLabel { color:#FFFFFF; }

            /* Section headers */
            .section-header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(214,181,106,0.18), stop:1 rgba(214,181,106,0.10));
                color:#E9D9A4; padding:14px 20px; font-size:12px; font-weight:700;
                font-family:'Montserrat','Segoe UI',Arial; letter-spacing:2px; text-transform:uppercase;
                border-left:4px solid #D6B56A; margin:12px 0 8px 0; border-radius:0 8px 8px 0;
            }

            /* Navigation buttons */
            .nav-button {
                text-align:left; padding:14px 18px; margin:6px 12px;
                border:none; border-radius:12px;
                background: rgba(255,255,255,0.10);
                color:#FFFFFF; font-size:14px; font-weight:500; font-family:'Open Sans','Segoe UI',Arial;
                transition: all 120ms ease-in-out;
            }
            .nav-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(214,181,106,0.28), stop:1 rgba(214,181,106,0.18));
                color:#FFFFFF; border:2px solid rgba(214,181,106,0.55);
            }
            .nav-button:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #D6B56A, stop:1 #E5C679);
                color:#0E1F3C; font-weight:700; border:2px solid #F0D68A;
            }

            /* Upload button */
            .upload-button {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3F7C7C, stop:1 #47908F);
                color:#FFFFFF; border:none; border-radius:12px; padding:16px 20px;
                margin:10px 14px; font-weight:700; font-size:14px; text-align:left;
            }
            .upload-button:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #47908F, stop:1 #57A3A2);
            }

            /* Generic buttons (content area) */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2C59C6, stop:1 #5A86F2);
                color:#FFFFFF; border:none; border-radius:10px; padding:10px 18px; font-weight:600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1F4BB5, stop:1 #4C79E8);
            }

            /* Inputs */
            QLineEdit, QTextEdit {
                background:#FDFEFE; color:#2A2E36;
                border:1px solid #D8DEE8; border-radius:10px; padding:10px; font-size:14px;
            }
            QLineEdit:focus, QTextEdit:focus { border:1px solid #2F6FDB; }

            /* Tables */
            QTableView { background:#FFFFFF; border:1px solid #E7EAF0; border-radius:12px; gridline-color:#E7EAF0; }
            QHeaderView::section { background:#0E1F3C; color:#E9D9A4; padding:10px; border:1px solid #29456E; font-weight:700; }

            /* Scrollbars */
            QScrollBar:vertical { border:none; background:transparent; width:10px; margin:0 2px; }
            QScrollBar::handle:vertical { background:#D6B56A; min-height:28px; border-radius:4px; }
            QScrollBar::handle:vertical:hover { background:#E5C679; }
            QScrollBar:horizontal { border:none; background:transparent; height:10px; margin:2px 0; }
            QScrollBar::handle:horizontal { background:#D6B56A; min-width:28px; border-radius:4px; }
            QScrollBar::handle:horizontal:hover { background:#E5C679; }

            /* Splitter */
            QSplitter::handle { background:#D6B56A; width:3px; }
            QSplitter::handle:hover { background:#E5C679; }
            """
        )

    @staticmethod
    def _stylesheet_dark() -> str:
        """Return dark theme with neon-cyber classy aesthetic."""
        return (
            """
            /* App base */
            QMainWindow { background:#0C0F17; color:#E3E8F5; }
            #mainContainer { background-color:#0C0F17; border-radius:12px; border:1px solid #242B3A; }

            /* Header: deep navy + electric gradient rail */
            #header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #141B2C, stop:1 #1A2440);
                border-bottom: 3px solid #5DA0FF;
                border-top-left-radius:12px; border-top-right-radius:12px; color:#E3E8F5;
            }
            #headerTitle { font-family:'Montserrat','Segoe UI',Arial; font-size:20px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#E3E8F5; }
            #timeLabel { font-family:'Open Sans','Segoe UI',Arial; font-size:13px; color:#88B4FF; font-weight:600; }
            #versionLabel { color:#6E7FA7; font-size:12px; }

            /* Content frame as neon card */
            #contentFrame {
                background:#121826;
                border-radius:16px;
                margin:12px;
                border:1px solid #28324A;
            }

            /* Sidebar with subtle luminous spine */
            #sidebar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #121A2B, stop:1 #0E1522);
                border-right: 3px solid #28324A;
                min-width:300px; max-width:300px; color:#E3E8F5;
            }
            #sidebar QLabel { color:#E3E8F5; }

            /* Section header: neon ribbon */
            .section-header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(93,160,255,0.20), stop:1 rgba(93,160,255,0.10));
                color:#AFC8FF; padding:14px 20px; font-size:12px; font-weight:700;
                font-family:'Montserrat','Segoe UI',Arial; letter-spacing:2px; text-transform:uppercase;
                border-left:4px solid #5DA0FF; margin:12px 0 8px 0; border-radius:0 8px 8px 0;
            }

            /* Navigation buttons: sleek chips */
            .nav-button {
                text-align:left; padding:14px 18px; margin:6px 12px;
                border:none; border-radius:12px;
                background:#1B2335; color:#F2F6FF;
                font-size:14px; font-weight:500; font-family:'Open Sans','Segoe UI',Arial;
                transition: all 120ms ease-in-out;
            }
            .nav-button:hover { background:#23304A; color:#FFFFFF; border:1px solid #5DA0FF; }
            .nav-button:pressed { background:#263554; color:#FFFFFF; }
            .nav-button:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #315FD9, stop:1 #5DA0FF);
                color:#FFFFFF; font-weight:700; border:1px solid #78B2FF;
            }

            /* Upload button: bright primary */
            .upload-button {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3F8CFF, stop:1 #2D78E3);
                color:#FFFFFF; border:none; border-radius:12px; padding:16px 20px; margin:10px 14px;
                font-weight:700; font-size:14px; text-align:left;
            }
            .upload-button:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2D78E3, stop:1 #1A66D1); }

            /* Generic content buttons */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2E57C8, stop:1 #4D7AF0);
                color:#FFFFFF; border:none; border-radius:10px; padding:10px 18px; font-weight:600;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #254CB8, stop:1 #3F6BE7); }

            /* Inputs */
            QLineEdit, QTextEdit {
                background:#0F1422; color:#E3E8F5;
                border:1px solid #2B3550; border-radius:10px; padding:10px; font-size:14px;
            }
            QLineEdit:focus, QTextEdit:focus { border:1px solid #5DA0FF; }

            /* Tables */
            QTableView { background:#121826; border:1px solid #28324A; border-radius:12px; gridline-color:#28324A; }
            QHeaderView::section { background:#0F1524; color:#AFC8FF; padding:10px; border:1px solid #28324A; font-weight:700; }

            /* Scrollbars: thin neon rails */
            QScrollBar:vertical { border:none; background:transparent; width:10px; margin:0 2px; }
            QScrollBar::handle:vertical { background:#355ACF; min-height:28px; border-radius:4px; }
            QScrollBar::handle:vertical:hover { background:#5DA0FF; }
            QScrollBar:horizontal { border:none; background:transparent; height:10px; margin:2px 0; }
            QScrollBar::handle:horizontal { background:#355ACF; min-width:28px; border-radius:4px; }
            QScrollBar::handle:horizontal:hover { background:#5DA0FF; }

            /* Splitter */
            QSplitter::handle { background:#355ACF; width:3px; }
            QSplitter::handle:hover { background:#5DA0FF; }
            """
        )

    # ──────────────────────── theme toggle
    def toggle_theme(self) -> None:
        """Toggle between light and dark themes and update UI."""
        self.current_theme = "dark" if getattr(self, "current_theme", "light") == "light" else "light"
        self._apply_stylesheet()

    # ──────────────────────── sidebar
    def _create_sidebar(self) -> QWidget:
        """Return fully-populated sidebar widget with modern styling."""

        sidebar_container = QWidget(objectName="sidebar")
        outer_layout = QVBoxLayout(sidebar_container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Enhanced logo section (without hotel emoji)
        logo_container = QFrame(objectName="logoContainer")
        logo_container.setMinimumHeight(90)
        logo_container.setStyleSheet("""
            #logoContainer {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(212, 175, 55, 0.15), stop:1 rgba(212, 175, 55, 0.05));
                border-bottom: 2px solid rgba(212, 175, 55, 0.3);
            }
        """)
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setSpacing(5)
        
        # Main title (without emoji)
        logo_lbl = QLabel("HOTEL ANALYTICS")
        logo_lbl.setStyleSheet("""
            font-size: 16pt;
            font-weight: 600;
            font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
            color: #FFFFFF;
            letter-spacing: 1.5px;
            text-align: center;
            margin-bottom: 2px;
        """)
        logo_lbl.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(logo_lbl)
        
        # Subtitle
        subtitle_lbl = QLabel("Premium Dashboard")
        subtitle_lbl.setStyleSheet("""
            font-size: 11px;
            font-family: 'Open Sans', Arial, sans-serif;
            color: #D4AF37;
            letter-spacing: 0.8px;
            text-align: center;
            font-style: italic;
        """)
        subtitle_lbl.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(subtitle_lbl)
        
        outer_layout.addWidget(logo_container)

        # Theme toggle button section
        theme_container = QFrame()
        theme_container.setStyleSheet("""
            background: rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid rgba(212, 175, 55, 0.2);
            padding: 10px;
        """)
        theme_layout = QHBoxLayout(theme_container)
        theme_layout.setContentsMargins(15, 10, 15, 10)
        
        # Theme toggle button
        self.theme_toggle_btn = QPushButton("🌙 Dark Mode")
        self.theme_toggle_btn.setObjectName("themeToggleButton")
        self.theme_toggle_btn.setStyleSheet("""
            QPushButton#themeToggleButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(212, 175, 55, 0.2), stop:1 rgba(212, 175, 55, 0.1));
                color: #D4AF37;
                border: 1px solid rgba(212, 175, 55, 0.4);
                border-radius: 8px;
                padding: 8px 15px;
                font-size: 12px;
                font-weight: 500;
                font-family: 'Open Sans', Arial, sans-serif;
            }
            QPushButton#themeToggleButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(212, 175, 55, 0.3), stop:1 rgba(212, 175, 55, 0.2));
                border: 1px solid rgba(212, 175, 55, 0.6);
            }
            QPushButton#themeToggleButton:pressed {
                background: rgba(212, 175, 55, 0.4);
            }
        """)
        self.theme_toggle_btn.clicked.connect(self.toggle_theme)
        theme_layout.addWidget(self.theme_toggle_btn)
        
        outer_layout.addWidget(theme_container)

        # Enhanced separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent, stop:0.5 #D4AF37, stop:1 transparent);
            margin: 0 20px;
            height: 2px;
        """)
        sep.setMaximumHeight(2)
        outer_layout.addWidget(sep)

        # scroll area for nav items
        scroll = QScrollArea(objectName="sidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """)

        scroll_content = QWidget()
        self.sidebar_layout = QVBoxLayout(scroll_content)
        self.sidebar_layout.setContentsMargins(0, 15, 0, 0)
        self.sidebar_layout.setSpacing(0)

        # DATA MANAGEMENT section with enhanced styling
        data_hdr = QLabel("📊 DATA MANAGEMENT")
        data_hdr.setProperty("class", "section-header")
        self.sidebar_layout.addWidget(data_hdr)

        upload_btn = QPushButton("📤  File Upload", objectName="uploadButton")
        upload_btn.setProperty("class", "upload-button")

        # ⬇️ Fix: pass *callable* to unified _show_view
        def _show_upload() -> None:
            self._show_view(views.file_upload_display)

        upload_btn.clicked.connect(_show_upload)
        self.sidebar_layout.addWidget(upload_btn)

        # Enhanced separator for sections
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent, stop:0.5 rgba(212, 175, 55, 0.3), stop:1 transparent);
            margin: 15px 20px;
            height: 1px;
        """)
        sep2.setMaximumHeight(1)
        self.sidebar_layout.addWidget(sep2)

        # load grouped routes
        self._load_views()
        self.sidebar_layout.addStretch()

        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

        return sidebar_container

    # ──────────────────────── demo data
    def _load_demo_data(self) -> None:
        """Load demo data from the Excel file in the data directory."""
        file_path = os.path.join("data", "hotel_data.xlsx")
        try:
            success = data.load_data(file_path)
            if success:
                df = data.get_dataframe()
                print(
                    f"Successfully loaded {file_path} with "
                    f"{len(df)} rows and {len(df.columns)} columns."
                )
                print("Columns:", df.columns.tolist())
                print("\nFirst few rows:")
                print(df.head())
            else:
                raise FileNotFoundError
        except Exception:
            # Fall back to synthetic data
            print("Could not load Excel file – using synthetic data instead.")
            rng = pd.date_range("2022-01-01", "2023-12-31")
            np.random.seed(42)
            demo = pd.DataFrame(
                {
                    "date": rng,
                    "room_type": np.random.choice(
                        ["Standard", "Deluxe", "Suite", "Executive"], len(rng)
                    ),
                    "occupancy": np.random.uniform(0.45, 0.9, len(rng)),
                    "rate": np.random.uniform(90, 260, len(rng)),
                }
            )
            demo["revpar"] = demo["rate"] * demo["occupancy"]
            data.load_dataframe(demo)

    # ──────────────────────── sidebar helpers
    def _load_views(self) -> None:
        hidden_groups = {"Advanced", "Data"}  # groups to hide from sidebar
        for group, routes in ROUTES_GROUPED.items():
            if group in hidden_groups:
                continue  # skip hidden groups
            self._add_view_group(group, routes)

    @staticmethod
    def _button_text(route: str) -> tuple[str, str]:
        """Return truncated label and professional icon."""
        max_len = 28
        trimmed = f"{route[: max_len - 3]}..." if len(route) > max_len else route
        lower = route.lower()
        
        # Font Awesome style icons for professional look
        if "revenue" in lower:
            icon = "💰  "
        elif "occupancy" in lower:
            icon = "🏨  "
        elif "booking" in lower or "reservation" in lower:
            icon = "📅  "
        elif "cancellation" in lower:
            icon = "❌  "
        elif "forecast" in lower:
            icon = "📊  "
        elif "customer" in lower or "guest" in lower:
            icon = "👥  "
        elif "pricing" in lower:
            icon = "💲  "
        elif "profitability" in lower:
            icon = "📈  "
        elif "performance" in lower or "kpi" in lower:
            icon = "⚡  "
        elif "dashboard" in lower or "overview" in lower:
            icon = "🏠  "
        elif "seasonality" in lower:
            icon = "📆  "
        elif "room cost" in lower or "cost" in lower:
            icon = "🏢  "
        elif "housekeeping" in lower:
            icon = "🧹  "
        elif "f&b" in lower or "food" in lower:
            icon = "🍽️  "
        elif "efficiency" in lower:
            icon = "⚙️  "
        elif "custom" in lower or "chart" in lower:
            icon = "📊  "
        elif "marketing" in lower:
            icon = "📢  "
        elif "what if" in lower or "scenario" in lower:
            icon = "🎯  "
        elif "turbo" in lower:
            icon = "⚡  "
        elif "operational" in lower or "operations" in lower:
            icon = "🔧  "
        elif "advanced" in lower or "ml" in lower:
            icon = "🤖  "
        elif "analytics" in lower:
            icon = "📊  "
        else:
            icon = "📈  "
        return trimmed, icon

    def _add_view_group(
        self,
        section: str,
        routes: dict[str, callable],  # type: ignore
    ) -> None:
        # Enhanced section headers with icons
        section_icons = {
            "Overview": "🏠",
            "Performance": "⚡",
            "Marketing": "📢",
            "Operations": "🔧",
            "Advanced": "🤖",
            "Data": "📊",
            "What If": "🎯",
            "Forecasting": "📊"
        }
        
        icon = section_icons.get(section, "📈")
        lbl = QLabel(f"{icon} {section.upper()}")
        lbl.setProperty("class", "section-header")
        self.sidebar_layout.addWidget(lbl)
        for name, factory in routes.items():
            self._add_nav_button(name, factory)

    def _add_nav_button(
        self,
        route: str,
        factory: callable,  # type: ignore
    ) -> None:
        display, icon = self._button_text(route)
        btn = QPushButton(f"{icon}{display}")
        btn.setProperty("class", "nav-button")
        btn.setCheckable(True)
        btn.setToolTip(f"View: {route}")
        btn.setStyleSheet("text-align:left;qproperty-wordWrap:true;")

        def _clicked() -> None:
            # uncheck siblings
            for i in range(self.sidebar_layout.count()):
                item = self.sidebar_layout.itemAt(i)
                w = item.widget() if item else None
                if isinstance(w, QPushButton) and w is not btn:
                    w.setChecked(False)
            btn.setChecked(True)
            self._show_view(factory)

        btn.clicked.connect(_clicked)
        self.sidebar_layout.addWidget(btn)

    @staticmethod
    def _button_text(route: str) -> tuple[str, str]:
        """Return truncated label and emoji icon."""
        max_len = 32
        trimmed = f"{route[: max_len - 3]}..." if len(route) > max_len else route
        lower = route.lower()
        if "revenue" in lower:
            icon = "💰 "
        elif "occupancy" in lower:
            icon = "🏨 "
        elif "booking" in lower:
            icon = "📅 "
        elif "cancellation" in lower:
            icon = "❌ "
        elif "forecast" in lower:
            icon = "📊 "
        elif "customer" in lower:
            icon = "👥 "
        elif "pricing" in lower:
            icon = "💲 "
        elif "profitability" in lower:
            icon = "📈 "
        elif any(k in lower for k in ("facilities", "usage")):
            icon = "🏢 "
        elif "upselling" in lower:
            icon = "⬆️ "
        elif any(k in lower for k in ("custom", "chart")):
            icon = "📊 "
        elif "marketing" in lower:
            icon = "📢 "
        else:
            icon = "📈 "
        return trimmed, icon

    # ──────────────────────── view switching
    def _show_view(self, target) -> None:
        """
        Display the requested view.

        `target` can be **either**:
        • a *callable* that returns a QWidget, **or**
        • a *string* key present in ``ROUTES_GROUPED``.
        """
        # Resolve the factory
        if callable(target):
            factory = target
        else:
            factory = ROUTES_GROUPED.get(target)
            if factory is None:
                raise ValueError(f"No view factory found for route: {target}")

        # Drop current widgets
        while self.content.count():
            w = self.content.widget(0)
            self.content.removeWidget(w)
            w.deleteLater()

        # Create / show the new widget
        try:
            view = factory()
            self.content.addWidget(view)
            shadow = QGraphicsDropShadowEffect()
            shadow.setColor(QColor(0, 0, 0, 0))
            shadow.setBlurRadius(20)
            view.setGraphicsEffect(shadow)
            self.animation = QPropertyAnimation(shadow, b"color", self)
            self.animation.setDuration(300)
            self.animation.setStartValue(QColor(0, 0, 0, 0))
            self.animation.setEndValue(QColor(0, 0, 120, 30))
            self.animation.setEasingCurve(QEasingCurve.InOutCubic)
            self.animation.start()
        except Exception as exc:  # pragma: no cover – visual fallback
            err = QLabel(f"❌ Unable to load view:\n{exc}")
            err.setAlignment(Qt.AlignCenter)
            err.setStyleSheet(
                "font-size:14pt;color:#FF6B6B;background:#2A232A;"
                "border-radius:10px;padding:20px;",
            )
            self.content.addWidget(err)
            traceback.print_exc()

    def _show_welcome(self) -> None:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        title = QLabel("Welcome to Hotel Analytics Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size:24pt;font-weight:bold;color:#4575DE;margin-bottom:20px;",
        )
        subtitle = QLabel(
            "Select a section from the sidebar to begin exploring your data",
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size:14pt;color:#8C98B9;")
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        self.content.addWidget(frame)

    # public helper
    def show_view(self, factory: callable) -> None:
        self._show_view(factory)

    def update_time(self) -> None:
        """Update the time display with Saudi Arabia time (UTC+03:00)."""
        current_utc = QDateTime.currentDateTimeUtc()
        saudi_time = current_utc.addSecs(3 * 60 * 60)
        time_string = saudi_time.toString("MMM dd, yyyy hh:mm:ss AP")
        self.time_label.setText(f"🕙 {time_string}")


# ──────────────────────────────── bootstrap


def _set_fallback_font(app: QApplication) -> None:
    """Select a reasonable fallback font when Roboto fails."""
    default_pt = 10
    db = QFontDatabase()
    chosen = ""
    if sys.platform == "darwin":
        for f in ("Helvetica Neue", "Arial", "Lucida Grande"):
            if f in db.families():
                chosen = f
                break
    elif sys.platform == "win32":
        for f in ("Segoe UI", "Tahoma", "Arial"):
            if f in db.families():
                chosen = f
                break
    else:  # linux / other
        for f in ("DejaVu Sans", "Noto Sans", "Ubuntu", "Arial"):
            if f in db.families():
                chosen = f
                break
    app.setFont(QFont(chosen, default_pt))
    print(f"Set fallback font to: {chosen or 'system default'} {default_pt}pt")


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")

        try:
            font_id = QFontDatabase.addApplicationFont(":/fonts/Roboto-Regular.ttf")
            if font_id == -1:
                raise RuntimeError("custom font not loaded")
            family = QFontDatabase.applicationFontFamilies(font_id)[0]
            app.setFont(QFont(family, 10))
            print(f"Loaded custom font: {family}")
        except Exception:  # pragma: no cover – visual
            print("Falling back to system font")
            _set_fallback_font(app)

        # Check for skip auth argument (for debugging) or if running as executable
        skip_auth = '--skip-auth' in sys.argv or getattr(sys, 'frozen', False)
        
        if skip_auth:
            print("Skipping authentication (executable mode or debug)")
            user = "executable_user"
        else:
            # Show authentication dialog before opening the dashboard
            try:
                user = authenticate_user()
                if not user:
                    # User cancelled or authentication failed
                    print("Authentication cancelled or failed")
                    sys.exit(0)
            except Exception as e:
                print(f"Authentication error: {e}")
                print("Falling back to no-auth mode")
                user = "fallback_user"

        print(f"User authenticated: {user}")
        window = MainWindow()
        window.show()
        print("Application started successfully")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Critical error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        input("Press Enter to exit...")  # Keep console open to see error
        sys.exit(1)
