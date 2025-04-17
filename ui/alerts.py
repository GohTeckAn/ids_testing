import tkinter as tk
from tkinter import ttk
import threading
import time
import pandas as pd

class AlertsWidget:
    def __init__(self, anomaly_detector):
        self.anomaly_detector = anomaly_detector
        self.root = tk.Tk()
        self.alerts_frame = create_alerts_window(self.root, anomaly_detector)
        self.alerts_frame.pack(padx=10, pady=10, fill='both', expand=True)

    def run(self):
        self.root.mainloop()

def create_alerts_window(parent, anomaly_detector):
    # Create alerts frame
    alerts_frame = ttk.Frame(parent)
    
    # Create alerts table
    alerts_table = create_alerts_table(alerts_frame, anomaly_detector)
    alerts_table.pack(padx=10, pady=10, fill='both', expand=True)
    
    # Start updating alerts periodically
    start_alerts_updates(alerts_frame, anomaly_detector)
    
    return alerts_frame

def create_alerts_table(parent, anomaly_detector):
    # Create treeview for alerts
    alerts_tree = ttk.Treeview(parent, columns=('Timestamp', 'Severity', 'Description'), show='headings')
    
    # Define column headings
    alerts_tree.heading('Timestamp', text='Timestamp')
    alerts_tree.heading('Severity', text='Severity')
    alerts_tree.heading('Description', text='Description')
    
    # Set column widths
    alerts_tree.column('Timestamp', width=150)
    alerts_tree.column('Severity', width=100)
    alerts_tree.column('Description', width=400)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(parent, orient='vertical', command=alerts_tree.yview)
    alerts_tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    
    # Store reference to update later
    parent.alerts_tree = alerts_tree
    
    return alerts_tree

def start_alerts_updates(alerts_frame, anomaly_detector):
    def update_alerts():
        while True:
            try:
                # Get current alerts
                alerts = anomaly_detector.get_alerts()
                
                # Clear existing alerts
                alerts_tree = alerts_frame.alerts_tree
                for i in alerts_tree.get_children():
                    alerts_tree.delete(i)
                
                # Add new alerts
                if alerts:
                    df = pd.DataFrame(alerts)
                    for _, alert in df.iterrows():
                        alerts_tree.insert('', 'end', values=(
                            alert.get('timestamp', 'N/A'),
                            alert.get('severity', 'N/A'),
                            alert.get('description', 'No description')
                        ))
                
                # Wait before next update
                time.sleep(5)
            except Exception as e:
                print(f"Alerts update error: {e}")
                break
    
    # Start update thread
    update_thread = threading.Thread(target=update_alerts, daemon=True)
    update_thread.start()
