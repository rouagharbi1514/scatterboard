#!/usr/bin/env python3
"""
Modern Authentication Module for Hotel Dashboard
================================================

Provides secure login functionality with a futuristic UI design.
Licensed software rental model with enhanced visual presentation.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QFrame, QApplication, QGraphicsDropShadowEffect
)
from PySide6.QtGui import QPixmap, QColor, QLinearGradient, QPainter, QPalette, QBrush


# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY LICENSE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Set these values when deploying to each hotel client

COMPANY_NAME = "Plaza"  # Hotel/Company name
LICENSE_USERNAME = "hamza"     # Username provided to hotel
LICENSE_PASSWORD = "ha055923"     # Password provided to hotel  
LICENSE_START_DATE = "2025-08-09"   # License start date (YYYY-MM-DD)
LICENSE_DURATION_DAYS = 365         # License duration in days (365 = 1 year)

# ═══════════════════════════════════════════════════════════════════════════════


class LicenseManager:
    """Manages company-controlled software licensing for hotel dashboard rental."""
    
    def __init__(self):
        self.company_name = COMPANY_NAME
        self.username = LICENSE_USERNAME
        self.password_hash = self._hash_password(LICENSE_PASSWORD)
        self.start_date = datetime.strptime(LICENSE_START_DATE, "%Y-%m-%d")
        self.end_date = self.start_date + timedelta(days=LICENSE_DURATION_DAYS)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> tuple[bool, str]:
        """
        Authenticate user credentials against company license.
        Returns (success, message)
        """
        # Check if username matches
        if username != self.username:
            return False, "Invalid username or password"
        
        # Check if password is correct
        if self._hash_password(password) != self.password_hash:
            return False, "Invalid username or password"
        
        # Check if license is still valid
        current_date = datetime.now()
        if current_date < self.start_date:
            return False, "License not yet active. Please contact support."
        
        if current_date > self.end_date:
            return False, "License has expired. Please contact support to renew."
        
        return True, "Login successful"
    
    def get_license_info(self) -> Dict[str, Any]:
        """Get license information including remaining days."""
        current_date = datetime.now()
        days_remaining = (self.end_date - current_date).days
        days_used = (current_date - self.start_date).days
        
        return {
            "company_name": self.company_name,
            "username": self.username,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "days_remaining": max(0, days_remaining),
            "days_used": max(0, days_used),
            "total_days": LICENSE_DURATION_DAYS,
            "is_active": self.start_date <= current_date <= self.end_date,
            "is_expired": current_date > self.end_date
        }


class LoginDialog(QDialog):
    """Modern futuristic login dialog for hotel dashboard authentication."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.license_manager = LicenseManager()
        self.authenticated_user = None
        self.setup_ui()
        self.apply_stylesheet()
        self.setup_animations()
    
    def setup_ui(self):
        """Setup the login dialog UI with modern design."""
        self.setWindowTitle("Hotel Analytics - Licensed Login")
        self.setMinimumSize(600, 700)
        self.resize(800, 800)
        self.setModal(True)
        self.setObjectName("loginDialog")
        
        # Set background image
        self.setAutoFillBackground(True)
        palette = self.palette()
        
        # Load background image
        background_pixmap = QPixmap("/Users/hamzamathlouthi/Desktop/Done/hotel-dashboard/background.jpg")
        if not background_pixmap.isNull():
            # Scale the background image to fit the dialog
            scaled_background = background_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatioByExpanding, 
                Qt.SmoothTransformation
            )
            palette.setBrush(QPalette.Window, QBrush(scaled_background))
        else:
            # Fallback to gradient if image can't be loaded
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor("#0c1e25"))
            gradient.setColorAt(1, QColor("#1a3a50"))
            palette.setBrush(QPalette.Window, QBrush(gradient))
        
        self.setPalette(palette)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        
        # Add top spacer
        layout.addStretch(1)
        
        # Header section
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(20)
        header_layout.setContentsMargins(30, 30, 30, 30)
        
        # Logo image instead of hotel emoji
        icon_label = QLabel()
        icon_label.setObjectName("iconLabel")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedSize(120, 120)
        
        # Load and set the logo image
        pixmap = QPixmap("/Users/hamzamathlouthi/Desktop/Done/hotel-dashboard/apple-touch-icon.png")
        if not pixmap.isNull():
            # Scale the image to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(scaled_pixmap)
        else:
            # Fallback text if image can't be loaded
            icon_label.setText("LOGO")
            icon_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        
        header_layout.addWidget(icon_label)
        
        # Title with glow effect
        title = QLabel("HOTEL ANALYTICS")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel(f"Licensed to: {self.license_manager.company_name}")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle)
        
        # Days left for license (replacing the license info fields)
        info = self.license_manager.get_license_info()
        days_left_label = QLabel(f"<b>Days Left:</b> {info['days_remaining']} days")
        days_left_label.setObjectName("licenseLabel")
        days_left_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(days_left_label)
        
        layout.addWidget(header_frame)
        
        # Add middle spacer
        layout.addStretch(1)
        
        # Form section with floating effect
        form_frame = QFrame()
        form_frame.setObjectName("formFrame")
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(40, 40, 40, 40)
        form_layout.setSpacing(30)
        
        # Username field (without emoji)
        username_container = QFrame()
        username_container.setObjectName("inputContainer")
        username_layout = QHBoxLayout(username_container)
        username_layout.setContentsMargins(20, 0, 20, 0)
        
        # Removed user emoji - no icon needed
        
        self.username_input = QLineEdit()
        self.username_input.setObjectName("inputField")
        self.username_input.setPlaceholderText("Enter your username")
        self.username_input.setMinimumHeight(60)
        self.username_input.setAlignment(Qt.AlignCenter)
        username_layout.addWidget(self.username_input)
        
        form_layout.addWidget(username_container)
        
        # Password field (without lock emoji)
        password_container = QFrame()
        password_container.setObjectName("inputContainer")
        password_layout = QHBoxLayout(password_container)
        password_layout.setContentsMargins(20, 0, 20, 0)
        
        # Removed lock emoji - no icon needed
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setObjectName("inputField")
        self.password_input.setMinimumHeight(60)
        self.password_input.setAlignment(Qt.AlignCenter)
        
        # Show/Hide password toggle button
        self.toggle_password_btn = QPushButton("👁")
        self.toggle_password_btn.setObjectName("togglePasswordButton")
        self.toggle_password_btn.setFixedSize(50, 50)
        self.toggle_password_btn.clicked.connect(self.toggle_password_visibility)
        
        password_layout.addWidget(self.password_input)
        password_layout.addWidget(self.toggle_password_btn)
        
        form_layout.addWidget(password_container)
        
        # Login button with animation
        self.login_button = QPushButton("ACCESS DASHBOARD")
        self.login_button.setObjectName("loginButton")
        self.login_button.setMinimumHeight(70)
        self.login_button.clicked.connect(self.handle_login)
        
        form_layout.addWidget(self.login_button)
        layout.addWidget(form_frame)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Add bottom spacer
        layout.addStretch(1)
        
        # Connect Enter key to login
        self.username_input.returnPressed.connect(self.handle_login)
        self.password_input.returnPressed.connect(self.handle_login)
        
        # Focus on username field
        self.username_input.setFocus()
        
        # Add shadow effects
        self.add_shadow_effects()
    
    def add_shadow_effects(self):
        """Add modern holographic shadow effects to UI elements."""
        # Header shadow
        header_shadow = QGraphicsDropShadowEffect()
        header_shadow.setBlurRadius(30)
        header_shadow.setXOffset(0)
        header_shadow.setYOffset(5)
        header_shadow.setColor(QColor(64, 164, 223, 150))
        self.findChild(QFrame, "headerFrame").setGraphicsEffect(header_shadow)
        
        # Form shadow
        form_shadow = QGraphicsDropShadowEffect()
        form_shadow.setBlurRadius(30)
        form_shadow.setXOffset(0)
        form_shadow.setYOffset(5)
        form_shadow.setColor(QColor(64, 164, 223, 150))
        self.findChild(QFrame, "formFrame").setGraphicsEffect(form_shadow)
        
        # Button shadow
        button_shadow = QGraphicsDropShadowEffect()
        button_shadow.setBlurRadius(15)
        button_shadow.setXOffset(0)
        button_shadow.setYOffset(5)
        button_shadow.setColor(QColor(64, 164, 223, 200))
        self.login_button.setGraphicsEffect(button_shadow)
    
    def setup_animations(self):
        """Setup subtle animations for UI elements."""
        # Button animation
        self.button_anim = QPropertyAnimation(self.login_button, b"geometry")
        self.button_anim.setDuration(300)
        self.button_anim.setEasingCurve(QEasingCurve.OutBack)
    
    def animate_button(self):
        """Animate the login button on click."""
        orig_rect = self.login_button.geometry()
        target_rect = orig_rect.adjusted(-5, -5, 5, 5)
        
        self.button_anim.setStartValue(orig_rect)
        self.button_anim.setEndValue(target_rect)
        self.button_anim.start()
    
    def toggle_password_visibility(self):
        """Toggle password visibility with animation."""
        # Animate the toggle button
        anim = QPropertyAnimation(self.toggle_password_btn, b"geometry")
        anim.setDuration(200)
        anim.setStartValue(self.toggle_password_btn.geometry().adjusted(-2, -2, 2, 2))
        anim.setEndValue(self.toggle_password_btn.geometry())
        anim.start()
        
        if self.password_input.echoMode() == QLineEdit.Password:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_btn.setText("🙈")
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_btn.setText("👁")
    
    def handle_login(self):
        """Handle login button click."""
        # Animate button
        self.animate_button()
        
        username = self.username_input.text().strip()
        password = self.password_input.text()
        
        if not username or not password:
            self.status_label.setText("Please enter both username and password.")
            return
        
        # Clear previous status
        self.status_label.setText("")
        
        # Authenticate using LicenseManager
        license_manager = LicenseManager()
        is_valid, message = license_manager.authenticate(username, password)
        
        if is_valid:
            self.authenticated_user = username
            self.accept()  # Close dialog with success
        else:
            self.status_label.setText(message)
            # Clear password field for security
            self.password_input.clear()
            self.password_input.setFocus()
    
    def apply_stylesheet(self):
        """Apply modern futuristic theme stylesheet."""
        self.setStyleSheet("""
            QDialog#loginDialog {
                background: transparent;
                border: none;
            }
            
            /* Header section */
            #headerFrame {
                background: rgba(10, 30, 50, 0.8);
                border-radius: 25px;
                border: 2px solid rgba(64, 164, 223, 0.5);
                backdrop-filter: blur(10px);
            }
            
            #iconLabel {
                background: qradialgradient(
                    cx:0.5, cy:0.5, radius: 0.5, fx:0.5, fy:0.5,
                    stop:0 rgba(64, 164, 223, 0.4),
                    stop:1 rgba(64, 164, 223, 0.1)
                );
                color: #40a4df;
                font-size: 60px;
                border-radius: 60px;
                border: 2px solid rgba(64, 164, 223, 0.5);
            }
            
            #titleLabel {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 36px;
                font-weight: 800;
                color: #ffffff;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-top: 15px;
                text-shadow: 
                    0 0 10px rgba(64, 164, 223, 0.8),
                    0 0 20px rgba(64, 164, 223, 0.6),
                    0 0 30px rgba(64, 164, 223, 0.4);
            }
            
            #subtitleLabel {
                font-size: 22px;
                color: #7fdbca;
                margin-top: 5px;
                font-weight: 600;
                letter-spacing: 1px;
                text-shadow: 0 0 10px rgba(127, 219, 202, 0.5);
            }
            
            #licenseLabel {
                color: #ffffff;
                font-size: 16px;
                font-weight: 500;
                background: rgba(20, 50, 80, 0.6);
                border-radius: 10px;
                padding: 10px;
                text-shadow: 0 0 5px rgba(64, 164, 223, 0.5);
            }
            
            /* Form section */
            #formFrame {
                background: rgba(15, 40, 60, 0.7);
                border-radius: 25px;
                border: 2px solid rgba(64, 164, 223, 0.4);
                backdrop-filter: blur(10px);
            }
            
            #inputContainer {
                background: rgba(10, 30, 50, 0.6);
                border-radius: 15px;
                border: 1px solid rgba(64, 164, 223, 0.3);
            }
            
            #inputField {
                background: transparent;
                color: #ffffff;
                font-size: 20px;
                font-weight: 500;
                border: none;
                padding: 15px;
                selection-background-color: #40a4df;
            }
            
            #inputField::placeholder {
                color: #a0c0d0;
                font-size: 18px;
                text-align: center;
            }
            
            #inputIcon {
                color: #40a4df;
                font-size: 30px;
                padding: 0 15px;
            }
            
            #togglePasswordButton {
                background: rgba(64, 164, 223, 0.2);
                color: #7fdbca;
                font-size: 24px;
                border: 2px solid rgba(64, 164, 223, 0.4);
                border-radius: 10px;
                padding: 10px;
            }
            
            #togglePasswordButton:hover {
                background: rgba(64, 164, 223, 0.4);
                color: #ffffff;
                border: 2px solid rgba(64, 164, 223, 0.6);
            }
            
            /* Login button */
            #loginButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #40a4df, stop:1 #2e86c1);
                color: #ffffff;
                border: none;
                border-radius: 15px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-weight: 700;
                font-size: 20px;
                text-transform: uppercase;
                letter-spacing: 2px;
                padding: 20px;
                transition: all 0.3s ease;
            }
            
            #loginButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #50b4ef, stop:1 #3e96d1);
                transform: translateY(-3px);
            }
            
            #loginButton:pressed {
                transform: translateY(1px);
            }
            
            /* Status label */
            #statusLabel {
                font-size: 16px;
                font-weight: 600;
                padding: 15px;
                border-radius: 10px;
                background: rgba(255, 107, 107, 0.3);
                color: #ffffff;
                text-shadow: 0 0 5px rgba(255, 107, 107, 0.8);
            }
        """)


def authenticate_user() -> Optional[str]:
    """
    Show login dialog and return authenticated username if successful.
    Returns None if login failed or was cancelled.
    """
    dialog = LoginDialog()
    if dialog.exec() == QDialog.Accepted:
        return dialog.authenticated_user
    return None