import nfstream
from nfstream import NFStreamer, NFPlugin
import pandas as pd

class My_Custom(NFPlugin):

    def on_init(self, packet, flow):
        if self.limit == 1:
            flow.expiration_id = -1

        flow.udps.payload_data = [] ## For Payload of each packet
        if packet.payload_size>0:
          flow.udps.payload_data.append(packet.ip_packet[-packet.payload_size:].hex()) ## First packet time
        else:
          flow.udps.payload_data.append(str('00'))
        
        flow.udps.delta_time = [] ## Delta_time column 
        flow.udps.delta_time.append(packet.delta_time)
        
        flow.udps.packet_direction =[] ## Packet Direction
        flow.udps.packet_direction.append(packet.direction)  ## Packet Direction
        
        flow.udps.ip_size = [] ## IP Packet Size
        flow.udps.ip_size.append(packet.ip_size) ## IP Packet Size
        
        flow.udps.transport_size = [] ## Transport Size
        flow.udps.transport_size.append(packet.transport_size) ## Transport Size
        
        flow.udps.payload_size = [] ## payload Size
        flow.udps.payload_size.append(packet.payload_size) ## payload Size
        
        ## Flags in each packet
        flow.udps.syn = [] ## Syn Flag Present 
        flow.udps.syn.append(packet.syn)
        
        flow.udps.cwr = [] ## CWR Flag Present 
        flow.udps.cwr.append(packet.cwr)
        
        flow.udps.ece = [] ## ECE Flag Present 
        flow.udps.ece.append(packet.ece)

        flow.udps.urg = [] ## URG Flag Present 
        flow.udps.urg.append(packet.urg)

        flow.udps.ack = [] ## ACK Flag Present 
        flow.udps.ack.append(packet.ack)

        flow.udps.psh = [] ## PSH Flag Present 
        flow.udps.psh.append(packet.psh)

        flow.udps.rst = [] ## RST Flag Present 
        flow.udps.rst.append(packet.rst)

        flow.udps.fin = [] ## RST Flag Present 
        flow.udps.fin.append(packet.fin)
               
        print("Flow_Initiated")
        

    def on_update(self, packet, flow):
        if packet.payload_size>0:
          flow.udps.payload_data.append(packet.ip_packet[-packet.payload_size:].hex()) ## Rest of the packet time
        else:
          flow.udps.payload_data.append(str('00'))
        
        flow.udps.delta_time.append(packet.delta_time)  ## Delta_time column  
        
        flow.udps.packet_direction.append(packet.direction)  ## Packet Direction
        
        flow.udps.ip_size.append(packet.ip_size) ## IP Packet Size
        
        flow.udps.transport_size.append(packet.transport_size) ## Transport Size
        
        flow.udps.payload_size.append(packet.payload_size) ## payload Size
        
        flow.udps.syn.append(packet.syn) ## Syn Flag Present        
        flow.udps.cwr.append(packet.cwr) ## CWR Flag Present        
        flow.udps.ece.append(packet.ece) ## ECE Flag Present        
        flow.udps.urg.append(packet.urg) ## URG Flag Present        
        flow.udps.ack.append(packet.ack) ## ACK Flag Present      
        flow.udps.psh.append(packet.psh) ## PSH Flag Present        
        flow.udps.rst.append(packet.rst) ## RST Flag Present       
        flow.udps.fin.append(packet.fin) ## FIN Flag Present 
         
        if self.limit == flow.bidirectional_packets:
           flow.expiration_id = -1 # -1 value force expiration
		

if __name__ == '__main__':  
    # pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/CommandInjection/CommandInjection.pcap'
    # NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    # print("*** Done Reading ***")
    # NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/CommandInjection.csv', columns_to_anonymize=[], flows_per_file=0)

    # pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/Backdoor_Malware/Backdoor_Malware.pcap'
    # NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    # print("*** Done Reading ***")
    # NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/Backdoor_Malware.csv', columns_to_anonymize=[], flows_per_file=0)    

    # pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/BrowserHijacking/BrowserHijacking.pcap'
    # NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    # print("*** Done Reading ***")
    # NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/BrowserHijacking.csv', columns_to_anonymize=[], flows_per_file=0)  

    # pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation.pcap'
    # NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    # print("*** Done Reading ***")
    # NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/DDoS-ACK_Fragmentation.csv', columns_to_anonymize=[], flows_per_file=0)      

    # pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation1.pcap'
    # NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    # print("*** Done Reading ***")
    # NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/DDoS-ACK_Fragmentation1.csv', columns_to_anonymize=[], flows_per_file=0)

    # pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation2.pcap'
    # NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    # print("*** Done Reading ***")
    # NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/DDoS-ACK_Fragmentation2.csv', columns_to_anonymize=[], flows_per_file=0)

    # pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation3.pcap'
    # NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    # print("*** Done Reading ***")
    # NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/DDoS-ACK_Fragmentation3.csv', columns_to_anonymize=[], flows_per_file=0)  

    # pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation4.pcap'
    # NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    # print("*** Done Reading ***")
    # NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/DDoS-ACK_Fragmentation4.csv', columns_to_anonymize=[], flows_per_file=0)
 
    pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/BenignTraffic/BenignTraffic.pcap'
    NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    print("*** Done Reading ***")
    NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/Benign.csv', columns_to_anonymize=[], flows_per_file=0) 
    
    pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/BenignTraffic/BenignTraffic1.pcap'
    NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    print("*** Done Reading ***")
    NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/Benign1.csv', columns_to_anonymize=[], flows_per_file=0)
    
    pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/BenignTraffic/BenignTraffic2.pcap'
    NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    print("*** Done Reading ***")
    NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/Benign2.csv', columns_to_anonymize=[], flows_per_file=0)
    
    
    pcap_file= 'F:/CIC IoT Dataset 2023/Packet_Level_Data/BenignTraffic/BenignTraffic3.pcap'
    NFS = NFStreamer(source=pcap_file ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    print("*** Done Reading ***") 
    NFS.to_csv(path='F:/CIC IoT Dataset 2023/Combined_Data/Benign3.csv', columns_to_anonymize=[], flows_per_file=0)
    


