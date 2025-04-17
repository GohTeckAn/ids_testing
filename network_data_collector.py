import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import csv
import logging
from collections import defaultdict
from scapy.all import sniff, IP, TCP, get_if_list, conf
import threading
import psutil
import socket

# Create directories
os.makedirs('data/logs', exist_ok=True)
os.makedirs('data/traffic', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Flow:
    def __init__(self, src_ip, dst_ip, src_port, dst_port):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        
        # Packet lengths
        self.fwd_lengths = []
        self.bwd_lengths = []
        self.fwd_header_lengths = []
        self.bwd_header_lengths = []
        
        # TCP specific
        self.init_win_bytes_forward = None
        self.subflow_fwd_bytes = 0
        self.subflow_bwd_bytes = 0
        
        # Segment sizes
        self.fwd_segments = []
        self.bwd_segments = []
        
        # Last update time
        self.last_update = time.time()

    def is_forward(self, packet):
        """Check if packet is in forward direction"""
        return (packet[IP].src == self.src_ip and 
                packet[IP].dst == self.dst_ip and 
                packet[TCP].sport == self.src_port and 
                packet[TCP].dport == self.dst_port)

    def add_packet(self, packet):
        """Add a packet to the flow"""
        try:
            if IP not in packet or TCP not in packet:
                return
            
            direction = 'forward' if self.is_forward(packet) else 'backward'
            length = len(packet)
            header_length = len(packet[IP]) + len(packet[TCP])
            segment_size = len(packet[TCP].payload)
            
            if direction == 'forward':
                self.fwd_lengths.append(length)
                self.fwd_header_lengths.append(header_length)
                self.fwd_segments.append(segment_size)
                self.subflow_fwd_bytes += length
                if self.init_win_bytes_forward is None:
                    self.init_win_bytes_forward = packet[TCP].window
            else:
                self.bwd_lengths.append(length)
                self.bwd_header_lengths.append(header_length)
                self.bwd_segments.append(segment_size)
                self.subflow_bwd_bytes += length
            
            self.last_update = time.time()
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")

    def get_stats(self):
        """Get flow statistics"""
        try:
            stats = {}
            
            # Basic port info
            stats['Destination Port'] = self.dst_port
            
            # Packet lengths
            if self.fwd_lengths:
                fwd_lengths = np.array(self.fwd_lengths)
                stats['Total Length of Fwd Packets'] = np.sum(fwd_lengths)
                stats['Fwd Packet Length Max'] = np.max(fwd_lengths)
                stats['Fwd Packet Length Mean'] = np.mean(fwd_lengths)
            else:
                logger.debug(f"No forward packets for flow {self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port}")
                stats.update({
                    'Total Length of Fwd Packets': 0,
                    'Fwd Packet Length Max': 0,
                    'Fwd Packet Length Mean': 0
                })
            
            if self.bwd_lengths:
                bwd_lengths = np.array(self.bwd_lengths)
                stats['Total Length of Bwd Packets'] = np.sum(bwd_lengths)
                stats['Bwd Packet Length Max'] = np.max(bwd_lengths)
                stats['Bwd Packet Length Mean'] = np.mean(bwd_lengths)
            else:
                logger.debug(f"No backward packets for flow {self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port}")
                stats.update({
                    'Total Length of Bwd Packets': 0,
                    'Bwd Packet Length Max': 0,
                    'Bwd Packet Length Mean': 0
                })
            
            # Header lengths
            stats['Fwd Header Length'] = np.mean(self.fwd_header_lengths) if self.fwd_header_lengths else 0
            stats['Fwd Header Length.1'] = np.sum(self.fwd_header_lengths) if self.fwd_header_lengths else 0
            
            # Average packet size
            all_lengths = self.fwd_lengths + self.bwd_lengths
            stats['Average Packet Size'] = np.mean(all_lengths) if all_lengths else 0
            
            # Segment sizes
            stats['Avg Fwd Segment Size'] = np.mean(self.fwd_segments) if self.fwd_segments else 0
            stats['Avg Bwd Segment Size'] = np.mean(self.bwd_segments) if self.bwd_segments else 0
            
            # Subflow bytes
            stats['Subflow Fwd Bytes'] = self.subflow_fwd_bytes
            stats['Subflow Bwd Bytes'] = self.subflow_bwd_bytes
            
            # Window size
            stats['Init_Win_bytes_forward'] = self.init_win_bytes_forward if self.init_win_bytes_forward else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating flow statistics: {e}")
            return None

class NetworkDataCollector:
    def __init__(self):
        # Initialize collector
        self.data_dir = 'data/traffic'
        self.logs_dir = 'data/logs'
        self.current_flows = defaultdict(list)
        self.collection_interval = 1.0  # seconds
        self.last_collection = time.time()
        self.flow_timeout = 60  # seconds
        
        # Start with new file
        self.create_new_file()
        logger.info("Network Data Collector initialized")
        
        # Packet processing lock
        self.lock = threading.Lock()

    @staticmethod
    def get_available_interfaces():
        """Get list of available network interfaces."""
        interfaces = []
        
        # Get network interfaces using psutil
        net_if_addrs = psutil.net_if_addrs()
        net_if_stats = psutil.net_if_stats()
        
        for iface_name in net_if_stats.keys():
            # Only include interfaces that are up
            if net_if_stats[iface_name].isup:
                # Get the interface addresses
                addrs = net_if_addrs.get(iface_name, [])
                # Try to find an IPv4 address for this interface
                ipv4 = next((addr.address for addr in addrs 
                           if addr.family == socket.AF_INET), "No IPv4")
                
                interfaces.append({
                    'name': iface_name,
                    'description': f"IPv4: {ipv4}"
                })
        
        return interfaces

    @staticmethod
    def choose_interface():
        """Interactive function to let user choose a network interface."""
        print("\nAvailable Network Interfaces:")
        print("-" * 80)
        print(f"{'ID':<4} {'Interface Name':<35} {'IPv4 Address'}")
        print("-" * 80)
        
        interfaces = NetworkDataCollector.get_available_interfaces()
        
        if not interfaces:
            print("No network interfaces found!")
            return None
            
        for idx, iface in enumerate(interfaces):
            print(f"{idx:<4} {iface['name']:<35} {iface['description']}")
        
        print("-" * 80)
        
        while True:
            try:
                choice = input("\nSelect interface ID: ").strip()
                choice = int(choice)
                if 0 <= choice < len(interfaces):
                    return interfaces[choice]['name']
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except IndexError:
                print("Invalid choice. Please try again.")

    def create_new_file(self):
        """Create a new CSV file with headers"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = os.path.join(self.data_dir, f'traffic_{timestamp}.csv')
        
        headers = [
            'Destination Port', 'Total Length of Fwd Packets',
            'Total Length of Bwd Packets', 'Fwd Packet Length Max',
            'Fwd Packet Length Mean', 'Bwd Packet Length Max',
            'Bwd Packet Length Mean', 'Fwd Header Length',
            'Average Packet Size', 'Avg Fwd Segment Size',
            'Avg Bwd Segment Size', 'Fwd Header Length.1',
            'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward'
        ]
        
        with open(self.current_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        logger.info(f"Created new file: {self.current_file}")

    def packet_callback(self, packet):
        """Process each captured packet"""
        try:
            if IP in packet and TCP in packet:
                with self.lock:
                    # Create flow key
                    forward_key = (
                        packet[IP].src, packet[IP].dst,
                        packet[TCP].sport, packet[TCP].dport
                    )
                    backward_key = (
                        packet[IP].dst, packet[IP].src,
                        packet[TCP].dport, packet[TCP].sport
                    )
                    
                    # Find or create flow
                    if forward_key in self.current_flows:
                        flow = self.current_flows[forward_key]
                    elif backward_key in self.current_flows:
                        flow = self.current_flows[backward_key]
                    else:
                        flow = Flow(*forward_key)
                        self.current_flows[forward_key] = flow
                    
                    # Add packet to flow
                    flow.add_packet(packet)
                    
        except Exception as e:
            logger.error(f"Error in packet callback: {e}")

    def cleanup_flows(self):
        """Remove expired flows"""
        current_time = time.time()
        with self.lock:
            expired = [
                key for key, flow in self.current_flows.items()
                if current_time - flow.last_update > self.flow_timeout
            ]
            for key in expired:
                del self.current_flows[key]
                logger.debug(f"Removed expired flow: {key}")

    def save_flow_stats(self):
        """Save current flow statistics"""
        try:
            with self.lock:
                for flow in self.current_flows.values():
                    stats = flow.get_stats()
                    if stats:
                        with open(self.current_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(list(stats.values()))
                        logger.debug(f"Saved flow stats for {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port}")
                    else:
                        logger.warning(f"Could not get stats for flow {flow.src_ip}:{flow.src_port} -> {flow.dst_ip}:{flow.dst_port}")
        except Exception as e:
            logger.error(f"Error saving flow stats: {e}")

    def run(self, interface=None, duration=None):
        """Run the collector with optional interface and duration."""
        try:
            # If no interface specified, let user choose
            if interface is None:
                interface = self.choose_interface()
                
            if interface is None:
                logger.error("No interface selected. Exiting.")
                return
            
            logger.info(f"Starting capture on interface: {interface}")
            print(f"Starting capture on interface: {interface}")
            
            # Start packet capture in a separate thread
            sniff_thread = threading.Thread(
                target=lambda: sniff(
                    iface=interface,
                    prn=self.packet_callback,
                    store=0,
                    timeout=duration
                )
            )
            sniff_thread.start()
            
            start_time = time.time()
            while True:
                current_time = time.time()
                
                # Check duration
                if duration and (current_time - start_time) >= duration:
                    break
                
                # Save stats and cleanup every interval
                if current_time - self.last_collection >= self.collection_interval:
                    self.save_flow_stats()
                    self.cleanup_flows()
                    self.last_collection = current_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Capture stopped by user")
        except Exception as e:
            logger.error(f"Error during capture: {e}")
        finally:
            # Save final stats
            self.save_flow_stats()
            logger.info("Capture finished")

if __name__ == "__main__":
    # Create collector
    collector = NetworkDataCollector()
    
    # Run with interface selection, default 60 seconds duration
    collector.run(duration=60)
