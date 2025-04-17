import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt
from ui.dashboard import DashboardWidget
from ui.alerts import AlertsWidget
from ui.network_monitor import NetworkMonitorWidget
from ui.sidebar import Sidebar
from modules.network_monitor import NetworkMonitor
from modules.model_manager import ModelManager
from modules.anomaly_detector import AnomalyDetector
import threading

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-based IDS")
        self.setMinimumSize(1200, 800)

        # Initialize backend components
        self.network_monitor = NetworkMonitor()
        self.model_manager = ModelManager()
        self.anomaly_detector = AnomalyDetector(self.model_manager)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create horizontal layout for sidebar and content
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create and setup sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # Create stacked widget for different pages
        self.stacked_widget = QStackedWidget()
        
        # Create pages
        self.dashboard = DashboardWidget(self.network_monitor, self.anomaly_detector)
        self.alerts = AlertsWidget(self.anomaly_detector)
        self.network_monitor_widget = NetworkMonitorWidget(self.network_monitor)

        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.dashboard)
        self.stacked_widget.addWidget(self.alerts)
        self.stacked_widget.addWidget(self.network_monitor_widget)

        # Add stacked widget to main layout
        main_layout.addWidget(self.stacked_widget)

        # Connect sidebar signals
        self.sidebar.dashboard_clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.dashboard))
        self.sidebar.alerts_clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.alerts))
        self.sidebar.network_monitor_clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.network_monitor_widget))

        # Start network monitoring in a separate thread
        self.start_monitoring()

    def start_monitoring(self):
        monitor_thread = threading.Thread(target=self.network_monitor.start_monitoring)
        monitor_thread.daemon = True
        monitor_thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
