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
        self._apply_stylesheet()

        # Start timer to update the time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)  # Update every second

    # ──────────────────────── stylesheet
    def _apply_stylesheet(self) -> None:
        """Apply a modern, dark-themed stylesheet."""

        self.setStyleSheet(
            # (long CSS omitted for brevity – identical to original)
            self._stylesheet(),
        )

    # pulled out to keep line-length sane
    @staticmethod
    def _stylesheet() -> str:
        """Return application-wide Qt CSS."""

        return (
            """
            /* Main container */
            #mainContainer { background-color:#12151C; border-radius:10px; }
            /* Header */
            #header { background:#1A1E27; border-bottom:1px solid #2E3444;
                border-top-left-radius:10px; border-top-right-radius:10px; }
            #headerTitle { font-family:'Segoe UI',Arial,sans-serif; font-size:18px;
                font-weight:bold; color:#E0E6F5; letter-spacing:2px; }
            #versionLabel { color:#5D6A85; font-size:12px; }
            #contentFrame { background:#171B24; border-radius:8px; margin:5px; }
            #sidebar { background:#1A1E27; border-right:1px solid #2E3444;
                min-width:280px; max-width:280px; }
            QLabel { color:#E0E6F5; }
            .section-header { background:linear-gradient(90deg,#2A3240,#1A1E27);
                color:#B8C5E3; padding:15px; font-size:12px; font-weight:bold;
                letter-spacing:1.5px; border-left:3px solid #4575DE;
                margin:8px 0 4px 0; border-radius:0 4px 4px 0; }
            /* Buttons */
            .nav-button { text-align:left; padding:14px 15px; margin:3px 10px;
                border:none; border-radius:8px; background:#2A3140; color:#F0F4F8;
                font-size:14px; font-weight:600; min-height:24px; }
            .nav-button:hover { background:#3A4553; color:#FFFFFF;
                border:1px solid #4575DE; transform:translateX(3px); }
            .nav-button:pressed { background:#4A5563; color:#FFFFFF;
                transform:translateX(1px); }
            .nav-button:checked { background:linear-gradient(90deg,#3563D9,#4575DE);
                color:#FFFFFF; font-weight:bold; border:1px solid #5A85F0; }
            .upload-button { background:linear-gradient(90deg,#3F8CFF,#2D78E3);
                color:#FFFFFF; border:none; border-radius:8px; padding:14px 15px;
                margin:8px 12px; font-weight:bold; font-size:14px; text-align:left;
                min-height:24px; }
            .upload-button:hover { background:linear-gradient(90deg,#2D78E3,#1A66D1);
                color:#FFFFFF; transform:translateY(-2px); }
            .upload-button:pressed { background:linear-gradient(90deg,#1A66D1,#0F5ABF);
                transform:translateY(0); }
            QScrollBar:vertical { border:none; background:#1A1E27; width:8px;
                border-radius:4px; }
            QScrollBar::handle:vertical { background:#3A4255; min-height:30px;
                border-radius:4px; }
            QScrollBar::handle:vertical:hover { background:#4575DE; }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical,
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background:none; height:0; width:0; }
            """
        )

    # ──────────────────────── sidebar
    def _create_sidebar(self) -> QWidget:
        """Return fully-populated sidebar widget."""

        sidebar_container = QWidget(objectName="sidebar")
        outer_layout = QVBoxLayout(sidebar_container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # logo
        logo_container = QFrame()
        logo_container.setMinimumHeight(80)
        logo_layout = QHBoxLayout(logo_container)
        logo_lbl = QLabel("HOTEL ANALYTICS")
        logo_lbl.setStyleSheet(
            "font-size:14pt;font-weight:bold;font-family:'Segoe UI';"
            "color:#E0E6F5;letter-spacing:1px;"
        )
        logo_lbl.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(logo_lbl)
        outer_layout.addWidget(logo_container)

        # separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background:#2E3444;margin:0 10px;")
        sep.setMaximumHeight(1)
        outer_layout.addWidget(sep)

        # scroll area for nav items
        scroll = QScrollArea(objectName="sidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollArea>QWidget>QWidget{background:transparent;}"
        )

        scroll_content = QWidget()
        self.sidebar_layout = QVBoxLayout(scroll_content)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(0)

        # DATA MANAGEMENT section
        data_hdr = QLabel("DATA MANAGEMENT")
        data_hdr.setProperty("class", "section-header")
        self.sidebar_layout.addWidget(data_hdr)

        upload_btn = QPushButton("📤 File Upload", objectName="uploadButton")
        upload_btn.setProperty("class", "upload-button")

        # ⬇️ Fix: pass *callable* to unified _show_view
        def _show_upload() -> None:
            self._show_view(views.file_upload_display)

        upload_btn.clicked.connect(_show_upload)
        self.sidebar_layout.addWidget(upload_btn)

        # separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background:#2E3444;margin:8px 10px;")
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
        for group, routes in ROUTES_GROUPED.items():
            self._add_view_group(group, routes)

    def _add_view_group(
        self,
        section: str,
        routes: dict[str, callable],  # type: ignore
    ) -> None:
        lbl = QLabel(section.upper())
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

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
