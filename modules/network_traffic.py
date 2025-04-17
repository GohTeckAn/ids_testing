import psutil
import time
from collections import defaultdict
import win32process
import win32gui
import win32con
import win32api
import os
import json
from datetime import datetime, timedelta

class NetworkTrafficMonitor:
    def __init__(self):
        self.previous_counters = {}
        self.app_connections = defaultdict(lambda: {'bytes_sent': 0, 'bytes_recv': 0})
        self.cumulative_data = self.load_cumulative_data()
        self.daily_data = self.load_daily_data()
        self.update_interval = 1.0  # seconds
        self.last_save_time = time.time()
        self.save_interval = 60  # Save every 60 seconds
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
    def load_cumulative_data(self):
        """Load cumulative network data from file"""
        data_file = os.path.join(os.path.dirname(__file__), 'network_data.json')
        try:
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    # Convert stored data back to defaultdict
                    return defaultdict(lambda: {'bytes_sent': 0, 'bytes_recv': 0},
                                    {k: v for k, v in data.items()})
            else:
                return defaultdict(lambda: {'bytes_sent': 0, 'bytes_recv': 0})
        except Exception as e:
            print(f"Error loading network data: {e}")
            return defaultdict(lambda: {'bytes_sent': 0, 'bytes_recv': 0})
    
    def save_cumulative_data(self):
        """Save cumulative network data to file"""
        data_file = os.path.join(os.path.dirname(__file__), 'network_data.json')
        try:
            # Convert defaultdict to regular dict for JSON serialization
            data_to_save = {k: v for k, v in self.cumulative_data.items()}
            with open(data_file, 'w') as f:
                json.dump(data_to_save, f)
        except Exception as e:
            print(f"Error saving network data: {e}")
    
    def load_daily_data(self):
        """Load daily network data from file"""
        data_file = os.path.join(os.path.dirname(__file__), 'daily_network_data.json')
        try:
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading daily network data: {e}")
            return {}
    
    def save_daily_data(self):
        """Save daily network data to file"""
        data_file = os.path.join(os.path.dirname(__file__), 'daily_network_data.json')
        try:
            with open(data_file, 'w') as f:
                json.dump(self.daily_data, f)
        except Exception as e:
            print(f"Error saving daily network data: {e}")
    
    def update_daily_data(self, current_traffic):
        """Update daily totals with current traffic data"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # If it's a new day, initialize the data
        if current_date not in self.daily_data:
            self.daily_data[current_date] = {
                'total_bytes_sent': 0,
                'total_bytes_recv': 0
            }
        
        # Add current traffic to daily totals
        for app_name, traffic in current_traffic.items():
            self.daily_data[current_date]['total_bytes_sent'] += traffic['bytes_sent']
            self.daily_data[current_date]['total_bytes_recv'] += traffic['bytes_recv']
        
        # Keep only last 30 days of data
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.daily_data = {k: v for k, v in self.daily_data.items() if k >= cutoff_date}
    
    def get_process_name(self, pid):
        try:
            process = psutil.Process(pid)
            try:
                return process.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return "Unknown"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "Unknown"
    
    def get_active_connections(self):
        """Get all active network connections and their associated processes"""
        connections = []
        
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED':
                pid = conn.pid
                if pid:
                    name = self.get_process_name(pid)
                    connections.append({
                        'pid': pid,
                        'name': name,
                        'local_addr': conn.laddr,
                        'remote_addr': conn.raddr,
                        'status': conn.status
                    })
        return connections
    
    def get_network_stats(self):
        """Get network statistics for all network interfaces"""
        current_time = time.time()
        current_counters = psutil.net_io_counters(pernic=True)
        stats = {}
        
        for interface, counters in current_counters.items():
            if interface in self.previous_counters:
                prev_counters = self.previous_counters[interface]
                time_diff = current_time - self.previous_counters.get(f"{interface}_time", current_time)
                
                bytes_sent = (counters.bytes_sent - prev_counters.bytes_sent) / time_diff
                bytes_recv = (counters.bytes_recv - prev_counters.bytes_recv) / time_diff
                
                # Update cumulative data
                if bytes_sent > 0 or bytes_recv > 0:  # Only update if there's actual traffic
                    stats[interface] = {
                        'bytes_sent': bytes_sent,
                        'bytes_recv': bytes_recv,
                        'total_sent': counters.bytes_sent,
                        'total_recv': counters.bytes_recv
                    }
            
            self.previous_counters[interface] = counters
            self.previous_counters[f"{interface}_time"] = current_time
        
        return stats
    
    def get_app_traffic(self):
        """Get network traffic statistics per application"""
        # Reset current counters
        self.app_connections.clear()
        
        # Get active connections
        connections = self.get_active_connections()
        
        # Get network stats
        net_stats = self.get_network_stats()
        
        # Calculate total current traffic
        total_bytes_sent = sum(stats['bytes_sent'] for stats in net_stats.values())
        total_bytes_recv = sum(stats['bytes_recv'] for stats in net_stats.values())
        
        # Count number of connections per application
        app_connection_counts = defaultdict(int)
        for conn in connections:
            app_connection_counts[conn['name']] += 1
        
        # Distribute traffic proportionally to number of connections
        total_connections = sum(app_connection_counts.values()) or 1
        current_traffic = {}
        for app_name, conn_count in app_connection_counts.items():
            traffic_ratio = conn_count / total_connections
            current_sent = total_bytes_sent * traffic_ratio
            current_recv = total_bytes_recv * traffic_ratio
            
            # Store current traffic for daily update
            current_traffic[app_name] = {
                'bytes_sent': current_sent,
                'bytes_recv': current_recv
            }
            
            # Update cumulative data
            self.cumulative_data[app_name]['bytes_sent'] += current_sent
            self.cumulative_data[app_name]['bytes_recv'] += current_recv
            
            self.app_connections[app_name] = {
                'bytes_sent': current_sent,
                'bytes_recv': current_recv,
                'total_sent': self.cumulative_data[app_name]['bytes_sent'],
                'total_recv': self.cumulative_data[app_name]['bytes_recv']
            }
        
        # Update daily totals
        self.update_daily_data(current_traffic)
        
        # Check if it's time to save data
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.save_cumulative_data()
            self.save_daily_data()
            self.last_save_time = current_time
        
        return dict(self.app_connections)
    
    def get_daily_totals(self):
        """Get daily network totals for the last 30 days"""
        return self.daily_data
    
    def format_bytes(self, bytes_value):
        """Format bytes into human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f}TB"
    
    def get_formatted_traffic(self):
        """Get formatted traffic data for display"""
        app_traffic = self.get_app_traffic()
        formatted_data = []
        
        for app_name, traffic in app_traffic.items():
            formatted_data.append({
                'name': app_name,
                'received': self.format_bytes(traffic['bytes_recv']),  # Current rate
                'sent': self.format_bytes(traffic['bytes_sent']),      # Current rate
                'total': self.format_bytes(traffic['total_recv'] + traffic['total_sent']),  # Cumulative
                'total_received': self.format_bytes(traffic['total_recv']),  # Cumulative received
                'total_sent': self.format_bytes(traffic['total_sent'])       # Cumulative sent
            })
        
        # Sort by total traffic
        formatted_data.sort(key=lambda x: float(x['total'][:-2]), reverse=True)
        return formatted_data
