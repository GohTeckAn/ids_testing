import tkinter as tk
from tkinter import ttk
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NetworkMonitorWidget:
    def __init__(self, network_monitor):
        self.network_monitor = network_monitor
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Network Monitor")
        
        # Create network monitor frame
        self.network_frame = create_network_monitor_window(self.root, network_monitor)
        self.network_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
    def run(self):
        self.root.mainloop()

def create_network_monitor_window(parent, network_monitor):
    # Create network monitor frame
    network_frame = ttk.Frame(parent)
    
    # Create network traffic table
    traffic_table = create_network_traffic_table(network_frame, network_monitor)
    traffic_table.pack(padx=10, pady=10, fill='both', expand=True)
    
    # Create network activity plot
    network_plot = create_network_activity_plot(network_frame, network_monitor)
    network_plot.get_tk_widget().pack(padx=10, pady=10, fill='both', expand=True)
    
    # Start updating network monitor periodically
    start_network_monitor_updates(network_frame, network_monitor)
    
    return network_frame

def create_network_traffic_table(parent, network_monitor):
    # Create treeview for network traffic
    traffic_tree = ttk.Treeview(parent, columns=('Application', 'Inbound', 'Outbound'), show='headings')
    
    # Define column headings
    traffic_tree.heading('Application', text='Application')
    traffic_tree.heading('Inbound', text='Inbound (KB)')
    traffic_tree.heading('Outbound', text='Outbound (KB)')
    
    # Set column widths
    traffic_tree.column('Application', width=200)
    traffic_tree.column('Inbound', width=100)
    traffic_tree.column('Outbound', width=100)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(parent, orient='vertical', command=traffic_tree.yview)
    traffic_tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    
    # Store reference to update later
    parent.traffic_tree = traffic_tree
    
    return traffic_tree

def create_network_activity_plot(parent, network_monitor):
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.set_title('Network Activity')
    ax.set_xlabel('Time')
    ax.set_ylabel('Traffic (KB)')
    
    # Embed plot in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    
    # Store references for updating
    parent.network_plot_fig = fig
    parent.network_plot_ax = ax
    parent.network_plot_canvas = canvas
    
    return canvas

def start_network_monitor_updates(network_frame, network_monitor):
    def update_network_monitor():
        while True:
            try:
                # Get current network stats
                stats = network_monitor.get_stats()
                
                # Update traffic table
                traffic_tree = network_frame.traffic_tree
                for i in traffic_tree.get_children():
                    traffic_tree.delete(i)
                
                if stats and 'applications' in stats:
                    for app, traffic in stats['applications'].items():
                        traffic_tree.insert('', 'end', values=(
                            app,
                            f"{traffic['inbound']/1024:.2f}",
                            f"{traffic['outbound']/1024:.2f}"
                        ))
                
                # Update network activity plot
                fig = network_frame.network_plot_fig
                ax = network_frame.network_plot_ax
                canvas = network_frame.network_plot_canvas
                
                # Clear previous plot
                ax.clear()
                
                # Plot new data if available
                if stats:
                    df = pd.DataFrame(stats)
                    df.plot(ax=ax)
                
                ax.set_title('Network Activity')
                ax.set_xlabel('Time')
                ax.set_ylabel('Traffic (KB)')
                
                # Redraw canvas
                canvas.draw()
                
                # Wait before next update
                time.sleep(2)
            except Exception as e:
                print(f"Network monitor update error: {e}")
                break
    
    # Start update thread
    update_thread = threading.Thread(target=update_network_monitor, daemon=True)
    update_thread.start()

# Example usage:
if __name__ == "__main__":
    # Create a network monitor widget
    network_monitor = NetworkMonitorWidget(None)  # Replace None with your network monitor object
    
    # Run the network monitor
    network_monitor.run()
