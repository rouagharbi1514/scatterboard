##!/usr/bin/env python3
# Design-only V5: no SSO / no Forgot, Aurora Night v2 background,
# glass card with gradient border, XL inputs, accessible focus, subtle animations.

import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QEasingCurve, QPropertyAnimation
from PySide6.QtGui import QPixmap, QColor, QPalette, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QGraphicsDropShadowEffect, QWidget, QGraphicsOpacityEffect
)

# ── Licence (inchangée)
COMPANY_NAME = "Plaza"
LICENSE_USERNAME = "hamza"
LICENSE_PASSWORD = "ha055923"
LICENSE_START_DATE = "2025-08-09"
LICENSE_DURATION_DAYS = 365


class LicenseManager:
    def __init__(self):
        self.username = LICENSE_USERNAME
        self.password_hash = hashlib.sha256(LICENSE_PASSWORD.encode()).hexdigest()
        self.start_date = datetime.strptime(LICENSE_START_DATE, "%Y-%m-%d")
        self.end_date = self.start_date + timedelta(days=LICENSE_DURATION_DAYS)

    def authenticate(self, u: str, p: str) -> Tuple[bool, str]:
        if u != self.username:
            return False, "Invalid username or password"
        if hashlib.sha256(p.encode()).hexdigest() != self.password_hash:
            return False, "Invalid username or password"
        now = datetime.now()
        if now < self.start_date:
            return False, "License not yet active. Please contact support."
        if now > self.end_date:
            return False, "License has expired. Please contact support to renew."
        return True, "Login successful"


class LoginDialog(QDialog):
    """
    UI modernisée V5 :
      • Fond “Aurora Night v2” (dégradés profonds + vignette douce)
      • Carte glass avec bordure dégradée premium
      • Zone Sign in épurée (sans SSO ni Forgot)
      • Inputs XL, focus ring accessible, animations subtiles
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.lic = LicenseManager()
        self.authenticated_user: Optional[str] = None
        self._build_ui()
        self._apply_qss()
        self._add_shadows()
        self._animate_title_underline()
        self._fade_in_card()

    # ── UI
    def _build_ui(self) -> None:
        self.setWindowTitle("Hotel Analytics – Login")
        self.setObjectName("loginDialog")
        self.setMinimumSize(900, 600)
        self.resize(1040, 640)
        self.setModal(True)

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#0B1220"))
        self.setPalette(pal)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)

        # — Gradient border wrapper (effet bordure moderne)
        self.cardWrap = QFrame(objectName="cardWrap")
        wrapLay = QVBoxLayout(self.cardWrap)
        wrapLay.setContentsMargins(1, 1, 1, 1)
        wrapLay.setSpacing(0)
        root.addWidget(self.cardWrap, 0, Qt.AlignHCenter | Qt.AlignVCenter)

        # Carte principale
        self.card = QFrame(objectName="card")
        self.card.setMaximumWidth(1100)
        wrapLay.addWidget(self.card)

        row = QHBoxLayout(self.card)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # ─ Pane gauche (hero / branding)
        left = QFrame(objectName="leftPane")
        left.setMinimumWidth(360)
        l = QVBoxLayout(left)
        l.setContentsMargins(32, 36, 32, 36)
        l.setSpacing(12)

        circle = QLabel(objectName="logoCircle")
        circle.setFixedSize(92, 92)
        circle.setAlignment(Qt.AlignCenter)
        pix = QPixmap(":/icons/app-icon.png")
        if not pix.isNull():
            circle.setPixmap(pix.scaled(58, 58, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            circle.setText("HD")
        l.addWidget(circle, 0, Qt.AlignLeft)

        brandTop = QLabel("Hotel Dashboard", objectName="brandTop")
        brandTop.setWordWrap(True)
        l.addWidget(brandTop)

        brandSub = QLabel("Premium Analytics • Plaza", objectName="brandSub")
        brandSub.setWordWrap(True)
        l.addWidget(brandSub)

        bp1 = QLabel("Real-time KPIs & RevPAR insights", objectName="bullet")
        bp2 = QLabel("Secure access • Enterprise-grade", objectName="bullet")
        l.addSpacing(6)
        l.addWidget(bp1)
        l.addWidget(bp2)
        l.addStretch(1)

        # Ruban séparateur
        sep = QFrame(objectName="separator")
        sep.setFixedWidth(24)

        # ─ Pane droite (Sign in épurée)
        right = QFrame(objectName="rightPane")
        r = QVBoxLayout(right)
        r.setContentsMargins(52, 42, 52, 42)
        r.setSpacing(16)

        form = QFrame(objectName="form")
        form.setMaximumWidth(520)
        f = QVBoxLayout(form)
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(12)

        headerRow = QVBoxLayout()
        title = QLabel("Sign in", objectName="formTitle")
        title.setAlignment(Qt.AlignLeft)
        self.accent = QFrame(objectName="accent")
        self.accent.setFixedHeight(3)
        self.accent.setMaximumWidth(64)
        headerRow.addWidget(title)
        headerRow.addWidget(self.accent)
        f.addLayout(headerRow)

        # Champs XL
        ulabel = QLabel("Username", objectName="fieldLabel")
        f.addWidget(ulabel)
        self.username_input = QLineEdit(placeholderText="Enter your username")
        self.username_input.setObjectName("lineEdit")
        self.username_input.setMinimumHeight(50)
        self.username_input.setClearButtonEnabled(True)
        f.addWidget(self.username_input)

        plabel = QLabel("Password", objectName="fieldLabel")
        f.addWidget(plabel)
        pwRow = QHBoxLayout()
        pwRow.setSpacing(8)
        self.password_input = QLineEdit(placeholderText="Enter your password")
        self.password_input.setObjectName("lineEdit")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(50)
        self.password_input.setClearButtonEnabled(True)
        self.eye = QPushButton("Show", objectName="eyeBtn")
        self.eye.setCursor(Qt.PointingHandCursor)
        self.eye.setFixedSize(72, 44)
        self.eye.clicked.connect(self._toggle_pw)
        pwRow.addWidget(self.password_input, 1)
        pwRow.addWidget(self.eye)
        f.addLayout(pwRow)

        # CTA principal
        self.login_button = QPushButton("Sign In", objectName="primaryBtn")
        self.login_button.setCursor(Qt.PointingHandCursor)
        self.login_button.setMinimumHeight(54)
        self.login_button.clicked.connect(self._handle_login)
        f.addWidget(self.login_button)

        # Hint & status
        hint = QLabel("Access is restricted to authorized personnel.", objectName="hint")
        f.addWidget(hint)
        self.status = QLabel("", objectName="status")
        self.status.setAlignment(Qt.AlignLeft)
        self.status.setVisible(False)
        f.addWidget(self.status)

        r.addWidget(form, 0, Qt.AlignLeft | Qt.AlignTop)
        row.addWidget(left)
        row.addWidget(sep)
        row.addWidget(right, 1)

        # UX
        self.username_input.returnPressed.connect(self._handle_login)
        self.password_input.returnPressed.connect(self._handle_login)
        self.setTabOrder(self.username_input, self.password_input)
        self.setTabOrder(self.password_input, self.login_button)
        self.username_input.setFocus()

        # Pop du CTA
        self.btn_anim = QPropertyAnimation(self.login_button, b"geometry", self)
        self.btn_anim.setDuration(220)
        self.btn_anim.setEasingCurve(QEasingCurve.OutCubic)

    # ── Styles
    def _apply_qss(self) -> None:
        self.setFont(QFont("Manrope", 10))
        self.setStyleSheet(r"""
            /* --- Background Aurora Night v2 + vignette --- */
            QDialog#loginDialog {
                background:
                  qradialgradient(cx:0.12, cy:0.10, radius:0.62, fx:0.12, fy:0.10,
                                  stop:0 #12275A, stop:1 transparent),
                  qradialgradient(cx:0.88, cy:0.90, radius:0.72, fx:0.88, fy:0.90,
                                  stop:0 #0E1F44, stop:1 transparent),
                  qconicalgradient(cx:0.62, cy:0.38, angle:210,
                                  stop:0 #0B1220, stop:0.35 #0D1630, stop:0.7 #0B1327, stop:1 #0B1220);
            }
            * { font-family:'Manrope','Inter','Segoe UI',Arial; }

            /* Gradient border wrapper */
            #cardWrap {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #4F7BFF, stop:0.5 #7EA6FF, stop:1 #C2D1FF);
                border-radius: 28px;
                padding: 0px;
            }

            /* Carte glass */
            #card {
                background: rgba(255,255,255,0.90);
                border: 1px solid rgba(180,195,230,0.30);
                border-radius: 27px;
            }

            /* Pane gauche – hero sombre premium */
            #leftPane {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #162550, stop:1 #1F3D8A);
                border-top-left-radius: 27px;
                border-bottom-left-radius: 27px;
            }
            #logoCircle {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #FFFFFF, stop:1 #F2F6FF);
                border:1px solid #E2E9F7;
                border-radius: 46px;
                color:#2242A4;
                font:800 18px 'Manrope';
            }
            #brandTop { color:#FFFFFF; font:800 24px 'Manrope'; letter-spacing:.2px; margin-top:12px; }
            #brandSub { color:#D6E7FF; font:600 13px 'Manrope'; }
            #bullet {
                color:#EAF1FF; font:600 12.5px 'Manrope';
                background: rgba(255,255,255,0.08);
                border:1px solid rgba(255,255,255,0.14);
                padding:8px 10px; border-radius:12px; margin-top:6px;
            }

            /* Séparateur ruban */
            #separator {
                background:#FFFFFF;
                border-left:1px solid #E5ECF9;
                border-right:1px solid #E5ECF9;
                border-top-right-radius:27px;
                border-bottom-right-radius:27px;
            }

            /* Pane droit translucide */
            #rightPane {
                background: rgba(255,255,255,0.50);
                border-top-right-radius:27px;
                border-bottom-right-radius:27px;
            }

            /* Titre & accent animé */
            #formTitle { color:#0F172A; font:800 26px 'Manrope'; margin-bottom:2px; }
            #accent {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2563EB, stop:1 #6A8BFF);
                border-radius: 2px;
            }

            /* Labels */
            #fieldLabel { color:#334155; font:700 12.5px 'Manrope'; margin-top:4px; }

            /* Inputs XL + focus ring accessible */
            #lineEdit {
                background:#FFFFFF;
                border:1px solid #D4E0F4;
                border-radius:14px;
                padding:12px 14px;
                font:600 14px 'Manrope';
                color:#0F172A;
                selection-background-color:#2563EB;
                selection-color:#FFFFFF;
            }
            #lineEdit::placeholder { color:#8FA1B3; font-weight:500; }
            #lineEdit:hover { border-color:#C6D4EF; }
            #lineEdit:focus {
                border:2px solid #2563EB;
                padding:11px 13px;
                box-shadow: 0 0 0 3px rgba(37,99,235,.14);
            }

            /* Show/Hide */
            #eyeBtn {
                background:#F4F7FC;
                border:1px solid #E3EAF6;
                border-radius:12px;
                color:#3E4C61;
                font:700 12.5px 'Manrope';
            }
            #eyeBtn:hover  { background:#EEF2FB; }
            #eyeBtn:pressed{ background:#E8EDFA; }

            /* CTA principal (pill) */
            #primaryBtn {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2563EB, stop:1 #1E40AF);
                color:#FFFFFF;
                border:none;
                border-radius:28px;
                font:800 15px 'Manrope';
                padding:14px 24px;
                margin-top:8px;
                letter-spacing:.2px;
            }
            #primaryBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2E6BFF, stop:1 #2444B3);
                box-shadow: 0 14px 28px rgba(37,99,235,.22);
            }
            #primaryBtn:pressed { background:#1B3A97; }

            /* Hint & status chips */
            #hint {
                color:#5B6B83;
                font:600 12px 'Manrope';
                margin-top:2px;
            }
            #status {
                color:#B91C1C;
                background:rgba(185,28,28,0.06);
                border:1px solid rgba(185,28,28,0.18);
                border-radius:12px;
                padding:8px 10px;
                font:700 12.5px 'Manrope';
                margin-top:8px;
            }
        """)

    def _add_shadows(self) -> None:
        sh = QGraphicsDropShadowEffect()
        sh.setBlurRadius(40)
        sh.setOffset(0, 22)
        sh.setColor(QColor(10, 22, 50, 46))
        self.card.setGraphicsEffect(sh)

    def _animate_title_underline(self) -> None:
        anim = QPropertyAnimation(self.accent, b"maximumWidth", self)
        anim.setDuration(680)
        anim.setStartValue(48)
        anim.setEndValue(180)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._accent_anim = anim

    def _fade_in_card(self) -> None:
        eff = QGraphicsOpacityEffect(self.cardWrap)
        self.cardWrap.setGraphicsEffect(eff)
        fade = QPropertyAnimation(eff, b"opacity", self)
        fade.setDuration(460)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.OutCubic)
        fade.start()
        self._fade_anim = fade

    # ── UX actions
    def _toggle_pw(self) -> None:
        if self.password_input.echoMode() == QLineEdit.Password:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.eye.setText("Hide")
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.eye.setText("Show")

    def _shake_card(self) -> None:
        g = self.card.geometry()
        anim = QPropertyAnimation(self.card, b"geometry", self)
        anim.setDuration(260)
        anim.setKeyValueAt(0.00, g)
        anim.setKeyValueAt(0.25, g.adjusted(6, 0, 6, 0))
        anim.setKeyValueAt(0.50, g.adjusted(-6, 0, -6, 0))
        anim.setKeyValueAt(0.75, g.adjusted(3, 0, 3, 0))
        anim.setKeyValueAt(1.00, g)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._shake_anim = anim

    def _handle_login(self) -> None:
        # Pop CTA
        g = self.login_button.geometry()
        self.btn_anim.stop()
        self.btn_anim.setStartValue(g)
        self.btn_anim.setEndValue(g.adjusted(-3, -2, 3, 2))
        self.btn_anim.finished.connect(lambda: self.login_button.setGeometry(g))
        self.btn_anim.start()

        u = self.username_input.text().strip()
        p = self.password_input.text()
        if not u or not p:
            self.status.setText("Please enter both username and password.")
            self.status.setVisible(True)
            self._shake_card()
            return

        ok, msg = self.lic.authenticate(u, p)
        if ok:
            self.authenticated_user = u
            self.accept()
        else:
            self.status.setText(msg)
            self.status.setVisible(True)
            self.password_input.clear()
            self.password_input.setFocus()
            self._shake_card()


def authenticate_user() -> Optional[str]:
    dlg = LoginDialog()
    return dlg.authenticated_user if dlg.exec() == QDialog.Accepted else None
