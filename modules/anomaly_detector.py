import pandas as pd
import numpy as np
from datetime import datetime
import csv
import os

class AnomalyDetector:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.alerts_file = 'data/alerts/alerts.csv'
        self.threshold = 0.8
        self.feature_columns = [
            'protocol', 'size', 'src_port', 'dst_port',
            'flow_duration', 'flow_packets', 'flow_bytes',
            'packets_per_second', 'bytes_per_second'
        ]
        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs(os.path.dirname(self.alerts_file), exist_ok=True)
        if not os.path.exists(self.alerts_file):
            with open(self.alerts_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'alert_type', 'severity', 'source', 
                               'destination', 'protocol', 'details'])

    def detect_anomalies(self, network_data):
        # Preprocess data
        try:
            processed_data = self._preprocess_data(network_data)
            
            # Get predictions from model
            predictions = self.model_manager.predict(processed_data)
            
            # Check for anomalies based on prediction scores
            for i, score in enumerate(predictions):
                if score > self.threshold:
                    self._log_alert(
                        alert_type='Anomalous Traffic',
                        severity='Medium' if score > 0.9 else 'Low',
                        source=network_data.iloc[i]['src_ip'],
                        destination=network_data.iloc[i]['dst_ip'],
                        protocol=network_data.iloc[i]['protocol'],
                        details=f'Anomaly score: {score:.2f}'
                    )
                    
            # Additional pattern-based detection
            self._analyze_patterns(network_data)
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")

    def _preprocess_data(self, data):
        # Create derived features
        data['flow_duration'] = pd.to_numeric(data['timestamp'].diff().dt.total_seconds(), errors='coerce')
        data['flow_packets'] = data.groupby(['src_ip', 'dst_ip'])['size'].transform('count')
        data['flow_bytes'] = data.groupby(['src_ip', 'dst_ip'])['size'].transform('sum')
        data['packets_per_second'] = data['flow_packets'] / data['flow_duration'].clip(lower=1)
        data['bytes_per_second'] = data['flow_bytes'] / data['flow_duration'].clip(lower=1)
        
        # Extract numeric features
        features = data[self.feature_columns].fillna(0)
        
        # Scale features
        scaled_features = (features - features.mean()) / features.std().clip(lower=1e-10)
        
        return scaled_features.values

    def _analyze_patterns(self, data):
        # Analyze for port scanning
        self._detect_port_scan(data)
        
        # Analyze for DoS
        self._detect_dos(data)
        
        # Analyze for data exfiltration
        self._detect_data_exfiltration(data)

    def _detect_port_scan(self, data):
        # Group by source IP and count unique destination ports
        port_counts = data.groupby('src_ip')['dst_port'].nunique()
        
        # Flag IPs scanning many ports
        for src_ip, port_count in port_counts.items():
            if port_count > 20:  # Threshold for number of unique ports
                self._log_alert(
                    'Port Scan Detected',
                    'Medium',
                    src_ip,
                    'Multiple',
                    'TCP',
                    f'Scanned {port_count} unique ports'
                )

    def _detect_dos(self, data):
        # Calculate packets per second for each source IP
        pps = data.groupby('src_ip')['size'].count() / 60  # packets per second
        
        # Flag high-rate sources
        for src_ip, rate in pps.items():
            if rate > 1000:  # Threshold for packets per second
                self._log_alert(
                    'Possible DoS Attack',
                    'High',
                    src_ip,
                    data[data['src_ip'] == src_ip]['dst_ip'].iloc[0],
                    'TCP',
                    f'Rate: {rate:.0f} packets/sec'
                )

    def _detect_data_exfiltration(self, data):
        # Calculate bytes per second for each destination
        bps = data.groupby('dst_ip')['size'].sum() / 60  # bytes per second
        
        # Flag high-volume destinations
        for dst_ip, volume in bps.items():
            if volume > 1_000_000:  # Threshold: 1 MB/s
                self._log_alert(
                    'Possible Data Exfiltration',
                    'High',
                    data[data['dst_ip'] == dst_ip]['src_ip'].iloc[0],
                    dst_ip,
                    'TCP',
                    f'Volume: {volume/1_000_000:.1f} MB/s'
                )

    def _log_alert(self, alert_type, severity, source, destination, protocol, details):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.alerts_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, alert_type, severity, source, 
                           destination, protocol, details])

    def get_alerts(self):
        if not os.path.exists(self.alerts_file):
            return []
        
        alerts = pd.read_csv(self.alerts_file)
        return alerts.to_dict('records')
