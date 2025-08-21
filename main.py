#!/usr/bin/env python3
"""
Hotel Dashboard – Main Application (Pro UI v3.1)
- Header: plus de bouton hamburger (≡)
- Sidebar: scrollbar masquée (défilement à la molette conservé)
- Thème: bouton Light/Dark en bas de la sidebar
- Aucune modification des routes/données
"""

from __future__ import annotations
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib  # noqa: F401
import os

from dotenv import load_dotenv
load_dotenv()

from PySide6.QtCore import (
    Qt, QEvent, QEasingCurve, QPropertyAnimation, QTimer, QDateTime
)
from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QScrollArea, QSplitter, QStackedWidget,
    QVBoxLayout, QWidget
)

from routes_grouped import ROUTES_GROUPED
import views
import data
import resources_rc  # noqa: F401
from auth import authenticate_user


# ────────────────────────── UI helpers

class CollapsibleSection(QWidget):
    """Section accordéon (header + corps), tolère mode compact (rail)."""
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self._collapsed = True

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(6, 2, 6, 2)
        self.root.setSpacing(4)

        self.header_btn = QPushButton(title)
        self.header_btn.setProperty("class", "section-header-btn")
        self.header_btn.setCheckable(True)
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.setToolTip(title)
        self.root.addWidget(self.header_btn)

        self.body = QFrame()
        self.body.setProperty("class", "section-body")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 4, 8, 8)
        self.body_layout.setSpacing(4)
        self.root.addWidget(self.body)

        self.anim = QPropertyAnimation(self.body, b"maximumHeight", self)
        self.anim.setDuration(170)
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)

        self.set_collapsed(True, animate=False)

    def add_subbutton(self, btn: QPushButton) -> None:
        self.body_layout.addWidget(btn)

    def buttons(self) -> list[QPushButton]:
        out: list[QPushButton] = []
        for i in range(self.body_layout.count()):
            w = self.body_layout.itemAt(i).widget()
            if isinstance(w, QPushButton):
                out.append(w)
        return out

    def set_collapsed(self, val: bool, animate: bool = True) -> None:
        self._collapsed = val
        self.header_btn.setChecked(not val)
        target = 0 if val else self.body.sizeHint().height()
        if animate:
            self.anim.stop()
            self.anim.setStartValue(self.body.maximumHeight())
            self.anim.setEndValue(target)
            self.anim.start()
        else:
            self.body.setMaximumHeight(target)

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_compact(self, compact: bool) -> None:
        self.header_btn.setProperty("compact", compact)
        self.header_btn.style().unpolish(self.header_btn)
        self.header_btn.style().polish(self.header_btn)


class Flyout(QWidget):
    """Panneau flottant (survol en mode rail compact)."""
    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setObjectName("flyout")
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 70))
        self.setGraphicsEffect(shadow)

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(10, 10, 10, 10)
        self.root.setSpacing(6)

        self.card = QFrame()
        self.card.setObjectName("flyoutCard")
        self.vbox = QVBoxLayout(self.card)
        self.vbox.setContentsMargins(10, 10, 10, 10)
        self.vbox.setSpacing(4)
        self.root.addWidget(self.card)

    def clear(self) -> None:
        while self.vbox.count():
            it = self.vbox.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)

    def populate(self, title: str, items: list[tuple[str, callable]]) -> None:
        self.clear()
        ttl = QLabel(title.upper())
        ttl.setObjectName("flyoutTitle")
        self.vbox.addWidget(ttl)
        for text, cb in items:
            btn = QPushButton(text)
            btn.setProperty("class", "sub-button")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, call=cb: (self.hide(), call()))
            self.vbox.addWidget(btn)

    def show_for(self, global_pos, width_hint: int = 260) -> None:
        self.adjustSize()
        g = self.geometry()
        self.setGeometry(global_pos.x(), global_pos.y(),
                         max(width_hint, g.width()), g.height())
        self.show()


# ────────────────────────── Main Window

class MainWindow(QMainWindow):
    """Primary window containing sidebar and stacked content views."""
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hotel Analytics Dashboard")
        self.setObjectName("MainWindow")

        # state
        self.sidebar_collapsed = False
        self.sections: list[CollapsibleSection] = []
        self.nav_buttons: list[QPushButton] = []
        self.active_btn: QPushButton | None = None
        self.flyout = Flyout(self)

        # geometry
        self.resize(1200, 800)
        self.setMinimumSize(960, 620)

        # root containers
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        main_container = QFrame(objectName="mainContainer")
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(0)
        vbox.addWidget(main_container)

        # ── Header (moderne, sans hamburger)
        self.header = QFrame(objectName="header")
        self.header.setFixedHeight(56)
        h = QHBoxLayout(self.header)
        h.setContentsMargins(12, 0, 12, 0)

        # ⛔ Bouton hamburger supprimé (non créé & non ajouté)
        title = QLabel("ScatterBoard", objectName="headerTitle")
        h.addWidget(title)
        h.addStretch()

        self.time_label = QLabel("", objectName="timeLabel")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self.time_label)
        h.addSpacing(10)

        version = QLabel("v3.0", objectName="versionLabel")
        version.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(version)

        header_shadow = QGraphicsDropShadowEffect()
        header_shadow.setBlurRadius(16)
        header_shadow.setOffset(0, 2)
        header_shadow.setColor(QColor(0, 0, 0, 22))
        self.header.setGraphicsEffect(header_shadow)
        main_layout.addWidget(self.header)

        # ── Splitter
        content_container = QFrame(objectName="contentContainer")
        content_layout = QHBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        main_layout.addWidget(content_container)

        self.split = QSplitter(Qt.Horizontal)
        content_layout.addWidget(self.split)

        self.sidebar = self._create_sidebar()

        self.content_frame = QFrame(objectName="contentFrame")
        cfl = QVBoxLayout(self.content_frame)
        cfl.setContentsMargins(18, 18, 18, 18)
        self.content = QStackedWidget()
        cfl.addWidget(self.content)

        self.split.addWidget(self.sidebar)
        self.split.addWidget(self.content_frame)
        self.split.setHandleWidth(2)
        self.split.setCollapsible(0, False)
        self.split.setCollapsible(1, False)
        self._apply_split_sizes()

        # data & welcome
        self._load_demo_data()
        self._show_welcome()

        # theme + clock
        self.current_theme = "light"
        self._apply_stylesheet()
        self._sync_theme_footer_label()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

    # ── Styles
    def _apply_stylesheet(self) -> None:
        css = self._stylesheet_dark() if self.current_theme == "dark" else self._stylesheet_light()
        self.setStyleSheet(css)

    def _sync_theme_footer_label(self) -> None:
        if hasattr(self, "theme_toggle_btn"):
            self.theme_toggle_btn.setText("☀  Light Mode" if self.current_theme == "dark" else "🌙  Dark Mode")

    @staticmethod
    def _stylesheet_light() -> str:
        return """
        /* Base */
        QMainWindow { background:#F6F8FC; color:#0F172A; }
        #mainContainer { background:#F6F8FC; border:0; }

        /* Header & Sidebar: gradient indigo pro */
        #header {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #203A91, stop:1 #2563EB);
            border:1px solid rgba(2,6,23,0.08);
            border-radius:16px;
            margin:10px 10px 8px 10px;
            color:#FFFFFF;
        }
        #headerTitle {
            font-family:'Inter','Poppins','Segoe UI','SF Pro Display',Arial,sans-serif;
            font-weight:600; font-size:18px; letter-spacing:.1px; color:#FFFFFF;
        }
        #timeLabel { font-family:'Inter','Segoe UI',Arial; font-weight:500; font-size:12px; color:#E7EEFF; }
        #versionLabel { color:#DDE7FF; font-size:12px; font-family:'Inter','Segoe UI',Arial; font-weight:500; }

        #contentFrame {
            background:#FFFFFF; border:1px solid #E6EAF1; border-radius:18px; margin:10px;
        }

        /* Sidebar container */
        #sidebar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #203A91, stop:1 #2855D7);
            border:1px solid rgba(2,6,23,0.08);
            color:#FFFFFF; padding:8px 8px 6px 8px;
            border-top-right-radius:28px; border-bottom-right-radius:28px;
            margin:8px 0 8px 8px;
        }
        #brandTitle {
            font-family:'Inter','Segoe UI',Arial;
            font-weight:700; font-size:16px; letter-spacing:.2px;
            color:#FFFFFF; padding:8px 10px 0 10px;
        }
        #brandSubtitle {
            font-family:'Inter','Segoe UI',Arial;
            font-weight:500; font-size:12px; letter-spacing:.1px;
            color:#D6E4FF; padding:0 10px 6px 10px;
        }

        /* Section header (btn) — plus doux */
        .section-header-btn {
            text-align:left;
            background:rgba(255,255,255,0.08);
            color:#FFFFFF;
            border:1px solid rgba(255,255,255,0.12);
            border-radius:12px;
            padding:9px 12px;
            margin:6px 6px 2px 6px;
            font-family:'Inter','Segoe UI',Arial;
            font-weight:600; font-size:12.5px; letter-spacing:.2px;
        }
        .section-header-btn:hover {
            background:rgba(255,255,255,0.14);
            border:1px solid rgba(255,255,255,0.18);
        }
        .section-header-btn:checked {
            background:#FFFFFF; color:#0F172A; border:1px solid rgba(2,6,23,0.06);
        }
        .section-header-btn[compact="true"] {
            min-width:44px; max-width:52px;
            padding:9px 8px; text-align:center; font-weight:700; letter-spacing:.3px;
        }

        .section-body { background:transparent; border:none; }

        /* Sub-buttons (routes) */
        .sub-button {
            text-align:left; padding:8px 12px; margin:3px 10px;
            border:1px solid transparent; border-radius:12px; background:transparent; color:#FFFFFF;
            font-family:'Inter','Open Sans','Segoe UI',Arial; font-weight:600; font-size:13px; letter-spacing:.1px;
        }
        .sub-button:hover { background:rgba(255,255,255,0.10); }
        .sub-button:checked {
            background:#FFFFFF; color:#0F172A;
            border-left:4px solid #203A91;
            border-top-left-radius:10px; border-bottom-left-radius:10px;
            border-top-right-radius:16px; border-bottom-right-radius:16px;
        }

        /* Upload CTA */
        .upload-button {
            background:rgba(255,255,255,0.14); color:#FFFFFF;
            border:1px solid rgba(255,255,255,0.20);
            border-radius:12px; padding:9px 12px; margin:6px 10px 2px 10px;
            font-weight:600; font-size:12.5px; text-align:left; font-family:'Inter','Segoe UI',Arial;
        }
        .upload-button:hover { background:rgba(255,255,255,0.20); }

        /* Footer sidebar */
        #sidebarFooter {
            background:rgba(255,255,255,0.08);
            border-top:1px solid rgba(255,255,255,0.16);
            border-bottom-right-radius:20px;
            padding:8px; margin:6px 6px 2px 6px;
        }
        #themeToggleButton {
            background:rgba(255,255,255,0.14); color:#FFFFFF;
            border:1px solid rgba(255,255,255,0.22);
            border-radius:10px; padding:8px 10px;
            font-family:'Inter','Segoe UI',Arial; font-weight:600; font-size:12.5px;
        }
        #themeToggleButton:hover { background:rgba(255,255,255,0.20); }

        /* Flyout */
        #flyout { background:transparent; }
        #flyoutCard { background:#FFFFFF; border:1px solid #E6EAF1; border-radius:14px; }
        #flyoutTitle {
            font-family:'Inter','Segoe UI',Arial; font-weight:700; font-size:12.5px;
            color:#0F172A; margin:0 0 6px 0; letter-spacing:.2px;
        }

        /* Scrollbars (masquées) & splitter */
        QScrollArea { border:none; background:transparent; }
        QScrollBar:vertical { width:0px; background:transparent; }
        QScrollBar::handle:vertical { background:transparent; }
        QScrollBar:horizontal { height:0px; background:transparent; }
        QScrollBar::handle:horizontal { background:transparent; }

        QSplitter::handle { background:#E6EAF1; width:2px; }
        QSplitter::handle:hover { background:#D5DCE8; }

        QPushButton {
            background:#335CFF; color:#FFFFFF; border:none;
            border-radius:10px; padding:10px 16px; font-weight:600;
            font-family:'Inter','Segoe UI',Arial;
        }
        QPushButton:hover { background:#274FE0; }
        """

    @staticmethod
    def _stylesheet_dark() -> str:
        return """
        QMainWindow { background:#0C111D; color:#E3E8F5; }
        #mainContainer { background:#0C111D; border:0; }

        #header {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #1C2E6A, stop:1 #1E3A8A);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:16px; margin:10px 10px 8px 10px; color:#E3E8F5;
        }
        #headerTitle { font-family:'Inter','Segoe UI',Arial; font-weight:600; font-size:18px; color:#E3E8F5; }
        #timeLabel { font-family:'Inter','Segoe UI',Arial; font-size:12px; color:#AFC8FF; font-weight:500; }
        #versionLabel { color:#8FA8E6; font-size:12px; font-family:'Inter','Segoe UI',Arial; font-weight:500; }

        #contentFrame { background:#101728; border-radius:18px; margin:10px; border:1px solid #263453; }

        #sidebar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1C2E6A, stop:1 #1E3A8A);
            border:1px solid rgba(255,255,255,0.08);
            color:#E3E8F5; padding:8px 8px 6px 8px;
            border-top-right-radius:28px; border-bottom-right-radius:28px;
            margin:8px 0 8px 8px;
        }
        #brandTitle { font-family:'Inter','Segoe UI',Arial; font-weight:700; font-size:16px; color:#FFFFFF; padding:8px 10px 0 10px; }
        #brandSubtitle { font-family:'Inter','Segoe UI',Arial; font-weight:500; font-size:12px; color:#DDE8FF; padding:0 10px 6px 10px; }

        .section-header-btn {
            text-align:left; background:rgba(255,255,255,0.08); color:#FFFFFF;
            border:1px solid rgba(255,255,255,0.10); border-radius:12px;
            padding:9px 12px; margin:6px 6px 2px 6px;
            font-family:'Inter','Segoe UI',Arial; font-weight:600; font-size:12.5px; letter-spacing:.2px;
        }
        .section-header-btn:hover { background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.16); }
        .section-header-btn:checked { background:#FFFFFF; color:#0F172A; border:1px solid rgba(255,255,255,0.10); }
        .section-header-btn[compact="true"] { min-width:44px; max-width:52px; padding:9px 8px; text-align:center; font-weight:700; letter-spacing:.3px; }

        .section-body { background:transparent; border:none; }

        .sub-button {
            text-align:left; padding:8px 12px; margin:3px 10px;
            border:1px solid transparent; border-radius:12px; background:transparent; color:#FFFFFF;
            font-size:13px; font-weight:600; letter-spacing:.1px; font-family:'Inter','Segoe UI',Arial;
        }
        .sub-button:hover { background:rgba(255,255,255,0.08); }
        .sub-button:checked {
            background:#FFFFFF; color:#0F172A;
            border-left:4px solid #5DA0FF;
            border-top-left-radius:10px; border-bottom-left-radius:10px;
            border-top-right-radius:16px; border-bottom-right-radius:16px;
        }

        #sidebarFooter { background:rgba(255,255,255,0.08); border-top:1px solid rgba(255,255,255,0.16); border-bottom-right-radius:20px; padding:8px; margin:6px 6px 2px 6px; }
        #themeToggleButton {
            background:rgba(255,255,255,0.14); color:#FFFFFF;
            border:1px solid rgba(255,255,255,0.22);
            border-radius:10px; padding:8px 10px; font-weight:600; font-size:12.5px; font-family:'Inter','Segoe UI',Arial;
        }
        #themeToggleButton:hover { background:rgba(255,255,255,0.20); }

        #flyout { background:transparent; }
        #flyoutCard { background:#101728; border:1px solid #263453; border-radius:14px; }
        #flyoutTitle { font-family:'Inter','Segoe UI',Arial; font-weight:700; font-size:12.5px; color:#DDE8FF; margin:0 0 6px 0; letter-spacing:.2px; }

        QScrollArea { border:none; background:transparent; }
        /* Scrollbars masquées */
        QScrollBar:vertical { width:0px; background:transparent; }
        QScrollBar::handle:vertical { background:transparent; }
        QScrollBar:horizontal { height:0px; background:transparent; }
        QScrollBar::handle:horizontal { background:transparent; }

        QSplitter::handle { background:#263453; width:2px; }
        QSplitter::handle:hover { background:#3550A6; }

        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2E57C8, stop:1 #4D7AF0);
            color:#FFFFFF; border:none; border-radius:10px; padding:10px 16px; font-weight:600;
            font-family:'Inter','Segoe UI',Arial;
        }
        QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #254CB8, stop:1 #3F6BE7); }
        """

    # ── Sidebar
    def _create_sidebar(self) -> QWidget:
        sidebar = QWidget(objectName="sidebar")
        sb_shadow = QGraphicsDropShadowEffect()
        sb_shadow.setBlurRadius(16)
        sb_shadow.setOffset(0, 2)
        sb_shadow.setColor(QColor(15, 23, 42, 28))
        sidebar.setGraphicsEffect(sb_shadow)

        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Branding (Hotel Dashboard)
        brand = QFrame()
        bl = QVBoxLayout(brand)
        bl.setContentsMargins(10, 8, 10, 4)
        title = QLabel("Hotel Dashboard", objectName="brandTitle")
        subtitle = QLabel("Premium Analytics", objectName="brandSubtitle")
        bl.addWidget(title)
        bl.addWidget(subtitle)
        outer.addWidget(brand)

        # Zone scrollable — scrollbar masquée, molette OK
        scroll = QScrollArea(objectName="sidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent}"
                             "QScrollArea> QWidget > QWidget{background:transparent}")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 6)
        content_layout.setSpacing(2)

        # Bloc DATA (upload)
        data_section = CollapsibleSection("DATA")
        upload_btn = QPushButton("File Upload", objectName="uploadButton")
        upload_btn.setProperty("class", "upload-button")
        upload_btn.setCursor(Qt.PointingHandCursor)
        upload_btn.clicked.connect(lambda: self._show_view(views.file_upload_display))
        data_section.add_subbutton(upload_btn)
        data_section.set_collapsed(True, animate=False)
        content_layout.addWidget(data_section)
        self.sections.append(data_section)

        # Sections depuis ROUTES_GROUPED (toutes visibles)
        for group, routes in ROUTES_GROUPED.items():
            section = CollapsibleSection(group.upper())
            for name, factory in routes.items():
                btn = QPushButton(self._trim(name))
                btn.setProperty("class", "sub-button")
                btn.setCheckable(True)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(name)
                btn._factory = factory  # type: ignore[attr-defined]

                def _click_factory(fab=factory, b=btn):
                    def _cb():
                        for other in self.nav_buttons:
                            if other is not b:
                                other.setChecked(False)
                        b.setChecked(True)
                        self.active_btn = b
                        self._show_view(fab)
                    return _cb

                btn.clicked.connect(_click_factory())
                section.add_subbutton(btn)
                self.nav_buttons.append(btn)

            # Accordéon exclusif
            def _on_header_click(s=section):
                def _cb():
                    if self.sidebar_collapsed:
                        self.sidebar_collapsed = False
                        self._apply_split_sizes()
                        for sec in self.sections:
                            sec.set_compact(False)
                    for sec in self.sections:
                        sec.set_collapsed(sec is not s, animate=True)
                return _cb

            section.header_btn.clicked.connect(_on_header_click())
            section.set_collapsed(True, animate=False)
            content_layout.addWidget(section)
            self.sections.append(section)

            # Survol pour flyout en rail
            section.header_btn.installEventFilter(self)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Footer – bouton thème
        footer = QFrame(objectName="sidebarFooter")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(6, 6, 6, 6)
        self.theme_toggle_btn = QPushButton("🌙  Dark Mode", objectName="themeToggleButton")
        self.theme_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.theme_toggle_btn.clicked.connect(self._toggle_theme_click)
        fl.addWidget(self.theme_toggle_btn)
        outer.addWidget(footer, 0)

        return sidebar

    # helpers
    @staticmethod
    def _trim(text: str, max_len: int = 28) -> str:
        return f"{text[:max_len-3]}..." if len(text) > max_len else text

    def _toggle_theme_click(self) -> None:
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self._apply_stylesheet()
        self._sync_theme_footer_label()

    # (Conservé pour compatibilité, plus lié à aucun bouton)
    def _toggle_sidebar(self) -> None:
        self.sidebar_collapsed = not self.sidebar_collapsed
        self._apply_split_sizes()
        for sec in self.sections:
            sec.set_compact(self.sidebar_collapsed)
            if self.sidebar_collapsed:
                sec.set_collapsed(True, animate=False)

    def _apply_split_sizes(self) -> None:
        rail = 64 if self.sidebar_collapsed else 232  # ultra-compact vs normal
        self.split.setSizes([rail, max(600, self.width() - rail)])

    # flyout au survol quand rail compact
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter and self.sidebar_collapsed:
            for sec in self.sections:
                if hasattr(sec, "header_btn") and obj is sec.header_btn:
                    items: list[tuple[str, callable]] = []
                    for b in sec.buttons():
                        fab = getattr(b, "_factory", None)
                        txt = b.text()
                        if fab:
                            items.append((txt, lambda f=fab: self._show_view(f)))
                    gpos = sec.header_btn.mapToGlobal(sec.header_btn.rect().topRight())
                    if items:
                        self.flyout.populate(sec.title, items)
                        self.flyout.show_for(gpos, width_hint=260)
                        break
        return super().eventFilter(obj, event)

    # ── Data / Views
    def _load_demo_data(self) -> None:
        file_path = os.path.join("data", "hotel_data.xlsx")
        try:
            success = data.load_data(file_path)
            if success:
                df = data.get_dataframe()
                print(f"Successfully loaded {file_path} with {len(df)} rows and {len(df.columns)} columns.")
                print("Columns:", df.columns.tolist())
                print("\nFirst few rows:")
                print(df.head())
            else:
                raise FileNotFoundError
        except Exception:
            print("Could not load Excel file – using synthetic data instead.")
            rng = pd.date_range("2022-01-01", "2023-12-31")
            np.random.seed(42)
            demo = pd.DataFrame({
                "date": rng,
                "room_type": np.random.choice(["Standard", "Deluxe", "Suite", "Executive"], len(rng)),
                "occupancy": np.random.uniform(0.45, 0.9, len(rng)),
                "rate": np.random.uniform(90, 260, len(rng)),
            })
            demo["revpar"] = demo["rate"] * demo["occupancy"]
            data.load_dataframe(demo)

    def _show_view(self, target) -> None:
        if callable(target):
            factory = target
        else:
            factory = ROUTES_GROUPED.get(target)
            if factory is None:
                raise ValueError(f"No view factory found for route: {target}")

        while self.content.count():
            w = self.content.widget(0)
            self.content.removeWidget(w)
            w.deleteLater()

        try:
            view = factory()
            self.content.addWidget(view)
            shadow = QGraphicsDropShadowEffect()
            shadow.setColor(QColor(0, 0, 0, 0))
            shadow.setBlurRadius(20)
            view.setGraphicsEffect(shadow)
            anim = QPropertyAnimation(shadow, b"color", self)
            anim.setDuration(240)
            anim.setStartValue(QColor(0, 0, 0, 0))
            anim.setEndValue(QColor(0, 0, 120, 30))
            anim.setEasingCurve(QEasingCurve.InOutCubic)
            anim.start()
        except Exception as exc:
            err = QLabel(f"❌ Unable to load view:\n{exc}")
            err.setAlignment(Qt.AlignCenter)
            err.setStyleSheet("font-size:14pt;color:#FF6B6B;background:#2A232A;border-radius:10px;padding:20px;")
            self.content.addWidget(err)
            traceback.print_exc()

    def _show_welcome(self) -> None:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        title = QLabel("Welcome to Hotel Analytics Dashboard")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size:22pt;font-weight:700;color:#2A56E8;margin-bottom:14px;"
            "font-family:'Inter','Segoe UI',Arial;"
        )
        subtitle = QLabel("Select a section from the sidebar to begin exploring your data")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size:13pt;color:#6B7280;")
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        self.content.addWidget(frame)

    def update_time(self) -> None:
        """Date/heure (Arabie Saoudite, UTC+03:00) en en-tête."""
        current_utc = QDateTime.currentDateTimeUtc()
        saudi_time = current_utc.addSecs(3 * 60 * 60)
        self.time_label.setText(saudi_time.toString("MMM dd, yyyy hh:mm:ss AP"))


# ────────────────────────── bootstrap

def _set_fallback_font(app: QApplication) -> None:
    default_pt = 10
    db = QFontDatabase()
    chosen = ""
    if sys.platform == "darwin":
        for f in ("Helvetica Neue", "Arial", "Lucida Grande"):
            if f in db.families():
                chosen = f; break
    elif sys.platform == "win32":
        for f in ("Segoe UI", "Tahoma", "Arial"):
            if f in db.families():
                chosen = f; break
    else:
        for f in ("DejaVu Sans", "Noto Sans", "Ubuntu", "Arial"):
            if f in db.families():
                chosen = f; break
    app.setFont(QFont(chosen or "Segoe UI", default_pt))
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
        except Exception:
            print("Falling back to system font")
            _set_fallback_font(app)

        skip_auth = "--skip-auth" in sys.argv or getattr(sys, "frozen", False)
        if skip_auth:
            print("Skipping authentication (executable mode or debug)")
            user = "executable_user"
        else:
            try:
                user = authenticate_user()
                if not user:
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
        input("Press Enter to exit...")
        sys.exit(1)

