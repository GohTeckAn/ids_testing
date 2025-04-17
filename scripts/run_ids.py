import os
import sys
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.network_monitor import NetworkMonitor
from modules.model_manager import ModelManager
from modules.anomaly_detector import AnomalyDetector

def create_dashboard_tab(notebook, network_monitor, model_manager, anomaly_detector):
    # Create dashboard frame
    dashboard_frame = ttk.Frame(notebook)
    
    # Overview section
    overview_label = ttk.Label(dashboard_frame, text="Network Overview", font=('Helvetica', 16, 'bold'))
    overview_label.pack(pady=10)
    
    # Packet count
    packet_count_label = ttk.Label(dashboard_frame, text="Packets Captured: 0")
    packet_count_label.pack(pady=5)
    dashboard_frame.packet_count_label = packet_count_label
    
    # Threat count
    threat_label = ttk.Label(dashboard_frame, text="Active Threats: 0")
    threat_label.pack(pady=5)
    dashboard_frame.threat_label = threat_label
    
    # Start updating dashboard
    start_dashboard_updates(dashboard_frame, network_monitor, anomaly_detector)
    
    return dashboard_frame

def start_dashboard_updates(dashboard_frame, network_monitor, anomaly_detector):
    def update_dashboard():
        while True:
            try:
                # Update packet count
                packet_count = network_monitor.packet_count
                dashboard_frame.packet_count_label.config(
                    text=f"Packets Captured: {packet_count}"
                )
                
                # Update threat count
                threats = anomaly_detector.get_alerts()
                dashboard_frame.threat_label.config(
                    text=f"Active Threats: {len(threats)}"
                )
                
                # Wait before next update
                time.sleep(2)
            except Exception as e:
                print(f"Dashboard update error: {e}")
                break
    
    # Start update thread
    update_thread = threading.Thread(target=update_dashboard, daemon=True)
    update_thread.start()

def create_alerts_tab(notebook, anomaly_detector):
    # Create alerts frame
    alerts_frame = ttk.Frame(notebook)
    
    # Alerts table
    alerts_tree = ttk.Treeview(alerts_frame, columns=('Timestamp', 'Severity', 'Description'), show='headings')
    alerts_tree.heading('Timestamp', text='Timestamp')
    alerts_tree.heading('Severity', text='Severity')
    alerts_tree.heading('Description', text='Description')
    alerts_tree.pack(padx=10, pady=10, fill='both', expand=True)
    alerts_frame.alerts_tree = alerts_tree
    
    # Start updating alerts
    start_alerts_updates(alerts_frame, anomaly_detector)
    
    return alerts_frame

def start_alerts_updates(alerts_frame, anomaly_detector):
    def update_alerts():
        while True:
            try:
                # Clear existing alerts
                alerts_tree = alerts_frame.alerts_tree
                for i in alerts_tree.get_children():
                    alerts_tree.delete(i)
                
                # Get and display alerts
                alerts = anomaly_detector.get_alerts()
                for alert in alerts:
                    alerts_tree.insert('', 'end', values=(
                        alert.get('timestamp', 'N/A'),
                        alert.get('severity', 'N/A'),
                        alert.get('description', 'No description')
                    ))

    # Create alerts tab
    alerts_frame = create_alerts_window(notebook, anomaly_detector)
    notebook.add(alerts_frame, text="Alerts")

    # Create network monitor tab
    network_monitor_frame = create_network_monitor_window(notebook, network_monitor)
    notebook.add(network_monitor_frame, text="Network Monitor")

    return root

def main():
    print("Starting AI-based Intrusion Detection System...")
    
    # Initialize components
    network_monitor = NetworkMonitor(simulation_mode=True)
    model_manager = ModelManager()
    anomaly_detector = AnomalyDetector(model_manager)

    # Start network monitoring in a separate thread
    monitor_thread = threading.Thread(target=network_monitor.start_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Create and run UI
    root = create_main_ui(network_monitor, model_manager, anomaly_detector)
    root.mainloop()

    # Cleanup
    try:
        network_monitor.stop_monitoring()
    except Exception as e:
        print(f"Error stopping network monitor: {e}")

if __name__ == "__main__":
    main()
