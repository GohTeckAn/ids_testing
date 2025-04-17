import tkinter as tk
from tkinter import ttk
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DashboardWidget:
    def __init__(self, network_monitor, anomaly_detector, model_manager):
        self.network_monitor = network_monitor
        self.anomaly_detector = anomaly_detector
        self.model_manager = model_manager
        self.system_running = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Network Monitoring Dashboard")
        
        # Create main dashboard frame
        self.dashboard_frame = ttk.Frame(self.root)
        self.dashboard_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Create sections for different dashboard components
        self.overview_section = self.create_overview_section(self.dashboard_frame, self.network_monitor)
        self.overview_section.pack(padx=10, pady=10, fill='x')
        
        self.threat_section = self.create_threat_section(self.dashboard_frame, self.anomaly_detector)
        self.threat_section.pack(padx=10, pady=10, fill='x')
        
        self.performance_section = self.create_performance_section(self.dashboard_frame, self.model_manager)
        self.performance_section.pack(padx=10, pady=10, fill='x')
        
        # Create network activity plot
        self.network_activity_plot = Figure(figsize=(6, 4), dpi=100)
        self.network_activity_ax = self.network_activity_plot.add_subplot(111)
        self.network_activity_ax.set_title('Network Activity')
        self.network_activity_ax.set_xlabel('Time')
        self.network_activity_ax.set_ylabel('Value')
        self.network_activity_canvas = FigureCanvasTkAgg(self.network_activity_plot, master=self.dashboard_frame)
        self.network_activity_canvas.draw()
        self.network_activity_canvas.get_tk_widget().pack(padx=10, pady=10, fill='x')
        
        # Start updating dashboard periodically
        self.start_dashboard_updates()
        
    def create_overview_section(self, parent, network_monitor):
        overview_frame = ttk.LabelFrame(parent, text="Network Overview")
        
        # Packet count
        packet_label = ttk.Label(overview_frame, text="Packets Captured: 0")
        packet_label.pack(padx=5, pady=5)
        
        # Store reference to update later
        overview_frame.packet_label = packet_label
        
        return overview_frame

    def create_threat_section(self, parent, anomaly_detector):
        threat_frame = ttk.LabelFrame(parent, text="Threat Detection")
        
        # Threat count
        threat_label = ttk.Label(threat_frame, text="Active Threats: 0")
        threat_label.pack(padx=5, pady=5)
        
        # Store reference to update later
        threat_frame.threat_label = threat_label
        
        return threat_frame

    def create_performance_section(self, parent, model_manager):
        performance_frame = ttk.LabelFrame(parent, text="Model Performance")
        
        # Model accuracy
        accuracy_label = ttk.Label(performance_frame, text="Model Accuracy: N/A")
        accuracy_label.pack(padx=5, pady=5)
        
        # Store reference to update later
        performance_frame.accuracy_label = accuracy_label
        
        return performance_frame

    def start_dashboard_updates(self):
        def update_dashboard():
            while True:
                try:
                    # Update packet count
                    packet_count = self.network_monitor.packet_count
                    self.overview_section.packet_label.config(
                        text=f"Packets Captured: {packet_count}"
                    )
                    
                    # Update threat count
                    threats = self.anomaly_detector.get_alerts()
                    self.threat_section.threat_label.config(
                        text=f"Active Threats: {len(threats)}"
                    )
                    
                    # Update model performance (placeholder)
                    self.performance_section.accuracy_label.config(
                        text="Model Accuracy: 95.5%"
                    )
                    
                    # Update network activity plot
                    stats = self.network_monitor.get_stats()
                    if stats:
                        df = pd.DataFrame(stats)
                        self.network_activity_ax.clear()
                        df.plot(ax=self.network_activity_ax)
                        self.network_activity_ax.set_title('Network Activity')
                        self.network_activity_ax.set_xlabel('Time')
                        self.network_activity_ax.set_ylabel('Value')
                        self.network_activity_canvas.draw()
                    
                    # Wait before next update
                    time.sleep(2)
                except Exception as e:
                    print(f"Dashboard update error: {e}")
                    break
        
        # Start update thread
        update_thread = threading.Thread(target=update_dashboard, daemon=True)
        update_thread.start()
        
    def run(self):
        self.root.mainloop()

# Example usage
if __name__ == "__main__":
    # Create network monitor, anomaly detector, and model manager instances
    network_monitor = NetworkMonitor()
    anomaly_detector = AnomalyDetector()
    model_manager = ModelManager()
    
    # Create dashboard
    dashboard = DashboardWidget(network_monitor, anomaly_detector, model_manager)
    
    # Run dashboard
    dashboard.run()
