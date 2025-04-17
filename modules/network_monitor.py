import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import csv
import logging
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Flow:
    def __init__(self, src_ip, dst_ip, src_port, dst_port):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        
        # Length statistics
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

    def add_packet(self, direction, packet_length, header_length, segment_size=None, init_win_size=None):
        if direction == 'forward':
            self.fwd_lengths.append(packet_length)
            self.fwd_header_lengths.append(header_length)
            if segment_size:
                self.fwd_segments.append(segment_size)
            if init_win_size and self.init_win_bytes_forward is None:
                self.init_win_bytes_forward = init_win_size
            self.subflow_fwd_bytes += packet_length
        else:
            self.bwd_lengths.append(packet_length)
            self.bwd_header_lengths.append(header_length)
            if segment_size:
                self.bwd_segments.append(segment_size)
            self.subflow_bwd_bytes += packet_length

    def get_stats(self):
        fwd_lengths = np.array(self.fwd_lengths) if self.fwd_lengths else np.array([0])
        bwd_lengths = np.array(self.bwd_lengths) if self.bwd_lengths else np.array([0])
        
        return {
            'Destination Port': self.dst_port,
            'Total Length of Fwd Packets': np.sum(fwd_lengths),
            'Total Length of Bwd Packets': np.sum(bwd_lengths),
            'Fwd Packet Length Max': np.max(fwd_lengths),
            'Fwd Packet Length Mean': np.mean(fwd_lengths),
            'Bwd Packet Length Max': np.max(bwd_lengths),
            'Bwd Packet Length Mean': np.mean(bwd_lengths),
            'Fwd Header Length': np.mean(self.fwd_header_lengths) if self.fwd_header_lengths else 0,
            'Average Packet Size': np.mean(np.concatenate([fwd_lengths, bwd_lengths])),
            'Avg Fwd Segment Size': np.mean(self.fwd_segments) if self.fwd_segments else 0,
            'Avg Bwd Segment Size': np.mean(self.bwd_segments) if self.bwd_segments else 0,
            'Fwd Header Length.1': np.sum(self.fwd_header_lengths) if self.fwd_header_lengths else 0,
            'Subflow Fwd Bytes': self.subflow_fwd_bytes,
            'Subflow Bwd Bytes': self.subflow_bwd_bytes,
            'Init_Win_bytes_forward': self.init_win_bytes_forward if self.init_win_bytes_forward else 0
        }

class HybridDetector:
    def detect(self, features):
        # This is a placeholder for your actual hybrid detector implementation
        # For demonstration purposes, it will always return a detection result
        return [
            {
                'is_anomaly': True,
                'anomaly_score': 0.8,
                'attack_predictions': {
                    'DDoS': [0.9, 0.1],
                    'PortScan': [0.1, 0.9]
                }
            }
        ]

class NetworkMonitor:
    def __init__(self, simulation_mode=True):
        self.simulation_mode = simulation_mode
        self.packet_count = 0
        self.flows = defaultdict(lambda: defaultdict(Flow))
        self.traffic_file = None
        self.current_traffic_data = []
        self.detector = None
        self._ensure_directories()
        self._load_detector()
        logger.info(f"NetworkMonitor initialized in {'simulation' if simulation_mode else 'capture'} mode")

    def _ensure_directories(self):
        os.makedirs('data/traffic', exist_ok=True)
        logger.info("Ensured traffic directory exists")

    def _load_detector(self):
        """Load the trained hybrid detector"""
        try:
            self.detector = HybridDetector()
            logger.info("Loaded hybrid detector successfully")
        except Exception as e:
            logger.error(f"Error loading detector: {e}")
            self.detector = None

    def analyze_traffic(self, flow_stats):
        """Analyze traffic using the hybrid detector"""
        if self.detector is None:
            logger.warning("No detector loaded, skipping analysis")
            return None
        
        # Prepare features in the correct order
        features = np.array([[
            flow_stats['Destination Port'],
            flow_stats['Total Length of Fwd Packets'],
            flow_stats['Total Length of Bwd Packets'],
            flow_stats['Fwd Packet Length Max'],
            flow_stats['Fwd Packet Length Mean'],
            flow_stats['Bwd Packet Length Max'],
            flow_stats['Bwd Packet Length Mean'],
            flow_stats['Fwd Header Length'],
            flow_stats['Average Packet Size'],
            flow_stats['Avg Fwd Segment Size'],
            flow_stats['Avg Bwd Segment Size'],
            flow_stats['Fwd Header Length.1'],
            flow_stats['Subflow Fwd Bytes'],
            flow_stats['Subflow Bwd Bytes'],
            flow_stats['Init_Win_bytes_forward']
        ]])
        
        # Get detection results
        results = self.detector.detect(features)
        
        if results and results[0]['is_anomaly']:
            logger.warning(f"Potential attack detected! Score: {results[0]['anomaly_score']:.2f}")
            if results[0]['attack_predictions']:
                for attack_type, probs in results[0]['attack_predictions'].items():
                    logger.warning(f"{attack_type.upper()} probability: {max(probs):.2f}")
        
        return results[0] if results else None

    def _create_new_traffic_file(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H')
        filename = f'data/traffic/traffic_{timestamp}.csv'
        
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Destination Port', 'Total Length of Fwd Packets',
                    'Total Length of Bwd Packets', 'Fwd Packet Length Max',
                    'Fwd Packet Length Mean', 'Bwd Packet Length Max',
                    'Bwd Packet Length Mean', 'Fwd Header Length',
                    'Average Packet Size', 'Avg Fwd Segment Size',
                    'Avg Bwd Segment Size', 'Fwd Header Length.1',
                    'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
                    'Init_Win_bytes_forward', 'Label'
                ])
        
        self.traffic_file = filename
        logger.info(f"Created new traffic file: {filename}")

    def _simulate_traffic(self):
        timestamp = datetime.now()
        
        # Generate random IPs and ports
        src_ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        dst_ip = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        src_port = random.randint(1024, 65535)
        dst_port = random.choice([80, 443, 22, 25, 53])
        
        # Get or create flow
        flow_key = (src_ip, dst_ip, src_port, dst_port)
        if flow_key not in self.flows:
            self.flows[flow_key] = Flow(src_ip, dst_ip, src_port, dst_port)
        
        # Simulate packet
        packet_length = random.randint(64, 1500)
        header_length = random.randint(20, 60)
        segment_size = packet_length - header_length
        init_win_size = random.randint(1000, 65535) if random.random() < 0.1 else None
        
        # Add packet to flow
        direction = random.choice(['forward', 'backward'])
        self.flows[flow_key].add_packet(
            direction, packet_length, header_length, 
            segment_size, init_win_size
        )
        
        # Get flow statistics
        stats = self.flows[flow_key].get_stats()
        
        # Analyze traffic
        detection_result = self.analyze_traffic(stats)
        if detection_result:
            stats['Label'] = 'DDoS' if detection_result['is_anomaly'] else 'BENIGN'
        else:
            stats['Label'] = 'BENIGN'
        
        # Save to current batch
        self.current_traffic_data.append(list(stats.values()))
        
        # Write to file every 10 packets
        if len(self.current_traffic_data) >= 10:
            self._write_traffic_data()
            logger.info(f"Wrote batch of {len(self.current_traffic_data)} packets")
            self.current_traffic_data = []

    def start_monitoring(self):
        logger.info("Starting network monitoring...")
        self._create_new_traffic_file()
        
        try:
            while True:
                if self.simulation_mode:
                    self._simulate_traffic()
                else:
                    self._capture_traffic()
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")

    def _write_traffic_data(self):
        if not self.current_traffic_data:
            return
            
        with open(self.traffic_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.current_traffic_data)

    def get_current_traffic(self):
        if os.path.exists(self.traffic_file):
            try:
                df = pd.read_csv(self.traffic_file)
                logger.info(f"Retrieved {len(df)} traffic records for analysis")
                return df
            except Exception as e:
                logger.error(f"Error reading traffic data: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
