from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIcon

class Sidebar(QWidget):
    # Define signals
    dashboard_clicked = pyqtSignal()
    alerts_clicked = pyqtSignal()
    network_monitor_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMaximumWidth(200)
        self.setMinimumWidth(200)
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-right: 1px solid #dee2e6;
            }
            QPushButton {
                text-align: left;
                padding: 10px;
                border: none;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QPushButton:checked {
                background-color: #dee2e6;
            }
            QLabel {
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
                color: #212529;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Add system name
        system_name = QLabel("AI-based IDS")
        layout.addWidget(system_name)

        # Add navigation buttons
        self.dashboard_btn = self.create_nav_button("Dashboard", "dashboard")
        self.alerts_btn = self.create_nav_button("Alerts", "alert")
        self.network_btn = self.create_nav_button("Network Monitor", "network")

        layout.addWidget(self.dashboard_btn)
        layout.addWidget(self.alerts_btn)
        layout.addWidget(self.network_btn)

        # Add stretch to push buttons to top
        layout.addStretch()

        self.setLayout(layout)

    def create_nav_button(self, text, icon_name):
        button = QPushButton(text)
        button.setCheckable(True)
        if icon_name == "dashboard":
            button.clicked.connect(self.dashboard_clicked.emit)
        elif icon_name == "alert":
            button.clicked.connect(self.alerts_clicked.emit)
        elif icon_name == "network":
            button.clicked.connect(self.network_monitor_clicked.emit)
        return button
