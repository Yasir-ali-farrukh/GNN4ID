import nfstream
from nfstream import NFStreamer, NFPlugin
import pandas as pd
import argparse
import os

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
               
        # print("Flow_Initiated")
        

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

    parser = argparse.ArgumentParser(description='Process pcap files.')
    parser.add_argument('pcap_files', help='List of pcap files to process.')
    parser.add_argument('Destination_path', help='The directory where the extracted flow+packet level information to be stored')
    args = parser.parse_args()
    pcap_files = args.pcap_files
    path_dir = args.Destination_path


    NFS = NFStreamer(source=pcap_files ,accounting_mode=1, idle_timeout=5, statistical_analysis=True, n_dissections=0, udps=My_Custom(limit=20))
    print("*** Done Reading ***")
    name = os.path.basename(pcap_files)
    name = name.split('.')[0]
    NFS.to_csv(path=path_dir+name+'.csv', columns_to_anonymize=[], flows_per_file=0)


    

    


