import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import sys
import os
import threading
import time

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.network_traffic import NetworkTrafficMonitor
from modules.model_manager import ModelManager
from ui.model_evaluation import ModelEvaluationPage

class IdsUI:
    def __init__(self):
        # Initialize network monitor
        self.network_monitor = NetworkTrafficMonitor()
        self.traffic_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = None
        self.max_points = 3600  # Store up to 1 hour of data
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Setup main window
        self.root = ctk.CTk()
        self.root.title("AI-Based IDS")
        self.root.geometry("1400x900")
        
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.content_frame = ctk.CTkFrame(self.root)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Initialize pages
        self.pages = {}
        self.create_dashboard_page()
        self.create_alerts_page()
        self.create_network_monitor_page()
        self.create_model_evaluation_page()
        
        # Show dashboard by default
        self.show_page("dashboard")
    
    def toggle_monitoring(self):
        self.monitoring_active = not self.monitoring_active
        if self.monitoring_active:
            self.start_time = time.time()
            self.traffic_history.clear()  # Clear previous history
            self.start_network_monitoring()
            self.system_switch.configure(text="Stop System")
        else:
            self.stop_network_monitoring()
            self.system_switch.configure(text="Start System")
    
    def start_network_monitoring(self):
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self.update_network_data, daemon=True)
            self.monitor_thread.start()
    
    def stop_network_monitoring(self):
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def update_network_data(self):
        while self.monitoring_active:
            try:
                # Get current network stats
                traffic_data = self.network_monitor.get_formatted_traffic()
                
                # Store in history
                self.traffic_history.append(traffic_data)
                if len(self.traffic_history) > self.max_points:
                    self.traffic_history.pop(0)
                
                # Update UI
                self.root.after(0, self.update_network_displays, traffic_data)
                
                # Wait before next update
                time.sleep(1)
            except Exception as e:
                print(f"Error updating network data: {e}")
                time.sleep(1)
    
    def create_sidebar(self):
        # Sidebar frame
        sidebar = ctk.CTkFrame(self.root, width=200)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(4, weight=1)  # Push buttons to top
        
        # System name
        system_name = ctk.CTkLabel(sidebar, text="AI-Based IDS", font=("Arial", 20, "bold"))
        system_name.grid(row=0, column=0, padx=20, pady=20)
        
        # Navigation buttons
        dashboard_btn = ctk.CTkButton(
            sidebar, text="Dashboard",
            command=lambda: self.show_page("dashboard")
        )
        dashboard_btn.grid(row=1, column=0, padx=20, pady=10)
        
        alerts_btn = ctk.CTkButton(
            sidebar, text="Alert",
            command=lambda: self.show_page("alerts")
        )
        alerts_btn.grid(row=2, column=0, padx=20, pady=10)
        
        network_btn = ctk.CTkButton(
            sidebar, text="Network Monitor",
            command=lambda: self.show_page("network")
        )
        network_btn.grid(row=3, column=0, padx=20, pady=10)
        
        #model_btn = ctk.CTkButton(
        #    sidebar, text="Model Evaluation",
        #    command=lambda: self.show_page("model")
        #)
        #model_btn.grid(row=4, column=0, padx=20, pady=10)
    
    def create_dashboard_page(self):
        page = ctk.CTkFrame(self.content_frame)
        
        # Title
        title = ctk.CTkLabel(page, text="Dashboard", font=("Arial", 24, "bold"))
        title.pack(anchor="w", padx=20, pady=20)
        
        # System start switch
        switch_frame = ctk.CTkFrame(page)
        switch_frame.pack(fill="x", padx=20, pady=10)
        
        switch_label = ctk.CTkLabel(switch_frame, text="System Status")
        switch_label.pack(side="left")
        
        self.system_switch = ctk.CTkButton(
            switch_frame,
            text="Start System",
            command=self.toggle_monitoring
        )
        self.system_switch.pack(side="left", padx=10)
        
        # Recent Alerts
        alerts_frame = ctk.CTkFrame(page)
        alerts_frame.pack(fill="x", padx=20, pady=10)
        
        alerts_title = ctk.CTkLabel(alerts_frame, text="Recent Alerts", font=("Arial", 16, "bold"))
        alerts_title.pack(anchor="w", pady=5)

        # Create a frame to hold both table and chart side by side
        alerts_content_frame = ctk.CTkFrame(alerts_frame)
        alerts_content_frame.pack(fill="x", pady=5)

        # Left side - Alert levels table
        table_frame = ctk.CTkFrame(alerts_content_frame)
        table_frame.pack(side="left", padx=(0, 10), fill="y")

        # Create table with alert levels
        columns = ("Level", "Count")
        self.alert_levels_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=2)
        
        for col in columns:
            self.alert_levels_table.heading(col, text=col)
            self.alert_levels_table.column(col, width=80)
        
        # Add sample data
        self.alert_levels_table.insert("", "end", values=("High", "10"))
        self.alert_levels_table.insert("", "end", values=("Low", "20"))
        self.alert_levels_table.pack(pady=5, padx=5)

        # Right side - Pie chart
        chart_frame = ctk.CTkFrame(alerts_content_frame)
        chart_frame.pack(side="left", fill="both", expand=True)

        # Create pie chart
        fig = plt.Figure(figsize=(2.5, 2.5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Sample data for pie chart
        sizes = [10, 20]  # Matching the table data
        labels = ['High', 'Low']
        colors = ['#FF6B6B', '#FFD93D']  # Red for High, Yellow for Low
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=5)

        # Network Activity
        activity_frame = ctk.CTkFrame(page)
        activity_frame.pack(fill="x", padx=20, pady=10)
        
        activity_title = ctk.CTkLabel(activity_frame, text="Network Activity", font=("Arial", 16, "bold"))
        activity_title.pack(anchor="w", pady=10)
        
        # Network activity table
        columns = ("Application", "Current Recv", "Current Sent")
        self.dashboard_network_table = ttk.Treeview(activity_frame, columns=columns, show="headings", height=3)
        
        for col in columns:
            self.dashboard_network_table.heading(col, text=col)
            self.dashboard_network_table.column(col, width=150)
        
        self.dashboard_network_table.pack(fill="x", pady=10)
        
        # Network activity graph
        fig = plt.Figure(figsize=(8, 3), dpi=100)
        self.network_ax = fig.add_subplot(111)
        self.network_canvas = FigureCanvasTkAgg(fig, master=activity_frame)
        self.network_canvas.draw()
        self.network_canvas.get_tk_widget().pack(fill="x", pady=10)
        
        self.pages["dashboard"] = page
        
    def create_alerts_page(self):
        page = ctk.CTkFrame(self.content_frame)
        
        # Title
        title = ctk.CTkLabel(page, text="Alert", font=("Arial", 24, "bold"))
        title.pack(anchor="w", padx=20, pady=20)
        
        # Alerts table
        columns = ("Name", "Levels", "Count", "Source", "Destination", "Protocol")
        alerts_table = ttk.Treeview(page, columns=columns, show="headings")
        
        for col in columns:
            alerts_table.heading(col, text=col)
            alerts_table.column(col, width=100)
        
        # Sample data
        alerts_table.insert("", "end", values=("APTs Alert", "High", "10", "192.168.2.124:4720", "192.168.1.1", "TCP"))
        alerts_table.insert("", "end", values=("DDoS Alert", "Low", "12", "192.168.2.124:8", "192.168.1.1", "ICMP"))
        alerts_table.insert("", "end", values=("Port Scanning Alert", "Low", "8", "192.168.2.124:4761", "192.168.1.1", "UDP"))
        
        alerts_table.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.pages["alerts"] = page
        
    def create_network_monitor_page(self):
        page = ctk.CTkFrame(self.content_frame)
        
        # Title
        title = ctk.CTkLabel(page, text="Network Monitor", font=("Arial", 24, "bold"))
        title.pack(anchor="w", padx=20, pady=20)
        
        # Date selector
        date_frame = ctk.CTkFrame(page)
        date_frame.pack(fill="x", padx=20, pady=10)
        
        date_label = ctk.CTkLabel(date_frame, text=datetime.now().strftime("%d %B %Y"))
        date_label.pack(side="left")
        
        # Network activity table
        columns = ("Application", "Current Recv", "Current Sent", "Total Recv", "Total Sent", "Total")
        self.network_table = ttk.Treeview(page, columns=columns, show="headings")
        
        for col in columns:
            self.network_table.heading(col, text=col)
            self.network_table.column(col, width=150)
        
        self.network_table.pack(fill="x", padx=20, pady=10)
        
        # Daily totals bar chart
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.daily_ax = fig.add_subplot(111)
        self.daily_canvas = FigureCanvasTkAgg(fig, master=page)
        self.daily_canvas.draw()
        self.daily_canvas.get_tk_widget().pack(fill="x", padx=20, pady=10)
        
        self.pages["network"] = page
        
    def create_model_evaluation_page(self):
        page = ModelEvaluationPage(self.content_frame, self.model_manager)
        self.pages["model"] = page
        
    def show_page(self, page_name):
        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()
        
        # Show selected page
        self.pages[page_name].pack(fill="both", expand=True)
    
    def update_network_displays(self, traffic_data):
        # Update dashboard network table and graph
        if "dashboard" in self.pages and hasattr(self, 'dashboard_network_table'):
            table = self.dashboard_network_table
            for item in table.get_children():
                table.delete(item)
            
            if self.monitoring_active:
                for app in traffic_data[:3]:  # Show top 3 apps
                    table.insert("", "end", values=(
                        app['name'],
                        app['received'],
                        app['sent']
                    ))
            
            # Update real-time graph in dashboard
            if hasattr(self, 'network_ax'):
                self.network_ax.clear()
                
                if self.monitoring_active and self.traffic_history:
                    # Calculate elapsed time for each point
                    current_time = time.time()
                    elapsed_seconds = current_time - self.start_time
                    
                    # Create time points (in seconds)
                    times = range(len(self.traffic_history))
                    
                    # Plot current traffic rates for top 3 apps
                    for app_idx, app in enumerate(traffic_data[:3]):
                        recv_values = []
                        sent_values = []
                        
                        for point in self.traffic_history:
                            app_data = next((x for x in point if x['name'] == app['name']), None)
                            if app_data:
                                # Convert current rates to numeric values
                                recv_value = float(app_data['received'][:-2])
                                sent_value = float(app_data['sent'][:-2])
                                recv_values.append(recv_value)
                                sent_values.append(sent_value)
                            else:
                                recv_values.append(0)
                                sent_values.append(0)
                        
                        if recv_values and sent_values:
                            # Plot received and sent rates separately
                            self.network_ax.plot(times, recv_values, 
                                               label=f"{app['name']} (Recv)", 
                                               linestyle='-')
                            self.network_ax.plot(times, sent_values, 
                                               label=f"{app['name']} (Sent)", 
                                               linestyle='--')
                    
                    # Set x-axis to show all points
                    if len(times) > 0:
                        self.network_ax.set_xlim(0, len(times))
                    
                    # Format timestamps on x-axis
                    def format_time(x, p):
                        seconds = int(x)
                        minutes = seconds // 60
                        remaining_seconds = seconds % 60
                        return f"{minutes:02d}:{remaining_seconds:02d}"
                    self.network_ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
                    
                    self.network_ax.set_title('Real-time Network Traffic Rates')
                    self.network_ax.set_ylabel('Traffic Rate')
                    self.network_ax.set_xlabel('Time (MM:SS)')
                    
                    # Use different line styles and colors for better distinction
                    colors = ['#FF9999', '#66B2FF', '#99FF99']  # Red, Blue, Green tints
                    for i, (line_recv, line_sent) in enumerate(zip(self.network_ax.lines[::2], self.network_ax.lines[1::2])):
                        line_recv.set_color(colors[i])
                        line_sent.set_color(colors[i])
                        line_sent.set_linestyle('--')
                    
                    # Add legend inside the graph
                    self.network_ax.legend(loc='upper right', 
                                         bbox_to_anchor=(0.98, 0.98),
                                         fancybox=True, 
                                         framealpha=0.8,
                                         fontsize='small')
                    
                    # Adjust layout
                    plt.tight_layout()
                
                self.network_canvas.draw()
        
        # Update network monitor page
        if "network" in self.pages and hasattr(self, 'network_table'):
            table = self.network_table
            for item in table.get_children():
                table.delete(item)
            
            if self.monitoring_active:
                for app in traffic_data:
                    table.insert("", "end", values=(
                        app['name'],
                        app['received'],
                        app['sent'],
                        app['total_received'],
                        app['total_sent'],
                        app['total']
                    ))
            
            # Update daily totals bar chart
            if hasattr(self, 'daily_ax'):
                self.daily_ax.clear()
                
                # Get daily totals from network monitor
                daily_data = self.network_monitor.get_daily_totals()
                
                # Sort dates and get last 7 days
                dates = sorted(daily_data.keys())[-7:]
                recv_traffic = []
                sent_traffic = []
                
                for date in dates:
                    data = daily_data[date]
                    recv_traffic.append(data['total_bytes_recv'])
                    sent_traffic.append(data['total_bytes_sent'])
                
                # Create bar chart
                x = range(len(dates))
                width = 0.35
                
                # Create bars for received and sent traffic
                self.daily_ax.bar([i - width/2 for i in x], recv_traffic, width, 
                                label='Inbound', color='lightblue')
                self.daily_ax.bar([i + width/2 for i in x], sent_traffic, width,
                                label='Outbound', color='lightgreen')
                
                # Format dates for x-axis
                date_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%d %b") for d in dates]
                self.daily_ax.set_xticks(x)
                self.daily_ax.set_xticklabels(date_labels, rotation=45, ha='right')
                
                self.daily_ax.set_title('Daily Network Traffic')
                self.daily_ax.set_ylabel('Traffic')
                self.daily_ax.legend()
                
                # Format y-axis values to readable sizes
                def format_bytes(x, p):
                    return self.network_monitor.format_bytes(x)
                
                self.daily_ax.yaxis.set_major_formatter(plt.FuncFormatter(format_bytes))
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                self.daily_canvas.draw()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = IdsUI()
    app.run()
