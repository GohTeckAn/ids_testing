from network_data_collector import NetworkDataCollector
import logging
import os
import time
from datetime import datetime

# Create directories
os.makedirs('data/logs', exist_ok=True)
os.makedirs('data/traffic', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_traffic(duration=60, interface=None):
    """Collect network traffic data for specified duration"""
    try:
        # Create collector
        collector = NetworkDataCollector()
        
        # Get WiFi interface if none specified
        if not interface:
            interface = "Qualcomm Atheros QCA9377 Wireless Network Adapter"
        
        # Start collection
        logger.info(f"Starting traffic collection on interface {interface} for {duration} seconds")
        collector.run(interface=interface, duration=duration)
        
        # Return path to collected data
        return collector.current_file
        
    except Exception as e:
        logger.error(f"Error during traffic collection: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Collect traffic for 5 minutes
    output_file = collect_traffic(duration=300)
    if output_file:
        logger.info(f"Traffic data saved to: {output_file}")
    else:
        logger.error("Failed to collect traffic data")
