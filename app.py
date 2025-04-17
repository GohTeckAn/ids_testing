from flask import Flask, render_template, jsonify
from modules.network_monitor import NetworkMonitor
from modules.anomaly_detector import AnomalyDetector
from modules.model_manager import ModelManager
import threading
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize components
network_monitor = NetworkMonitor()
model_manager = ModelManager()
anomaly_detector = AnomalyDetector(model_manager)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/network-monitor')
def network_monitor_page():
    return render_template('network_monitor.html')

@app.route('/api/network-stats')
def get_network_stats():
    return jsonify(network_monitor.get_stats())

@app.route('/api/alerts')
def get_alerts():
    return jsonify(anomaly_detector.get_alerts())

if __name__ == '__main__':
    # Start network monitoring in a separate thread
    monitor_thread = threading.Thread(target=network_monitor.start_monitoring)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
