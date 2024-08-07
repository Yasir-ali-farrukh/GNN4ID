import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

def additional_features(file_name,window_size=350,http_ports = [443, 8080,80],
                       vulnerable_ports = [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080],
                       dns_ports=[53]):
    try:
        data=pd.read_csv(file_name)
    except:
        print("file reading error")
        return ""
    try:
        data = data.sort_values(by='src2dst_first_seen_ms')
    except:
        print (" CSV does not contain src2dst_first_seen_ms for initial sorting ")
        return ""

    #Supporting Functions
    def udp_requests_in_window(group, window_size):
        return group.rolling(window=window_size, min_periods=1).sum()
    def tcp_requests_in_window(group, window_size):
        return group.rolling(window=window_size, min_periods=1).sum()
    def syn_packets_in_window(group, window_size):
        return group.rolling(window=window_size, min_periods=1).sum()
    def unique_ports_in_window(group, window_size):
        result = group.rolling(window=window_size, min_periods=1).apply(lambda x: len(set(x)), raw=True)
        return result
    def icmp_requests_in_window(group, window_size):
        return group.rolling(window=window_size, min_periods=1).sum()
    def dur_packets_in_window(group, window_size):
        return group.rolling(window=window_size, min_periods=1).mean()
        
    # Initialize LabelEncoder for reference features
    label_encoder = LabelEncoder()
    label_encoder2 = LabelEncoder()
    data['src_dst_ip'] = data['src_ip'] + '-' + data['dst_ip']
    data['src_dst_encoded'] = label_encoder.fit_transform(data['src_dst_ip'])
    data['dst_ip_encoded']=label_encoder2.fit_transform(data['dst_ip'])


    
    #Calcualting different features
    data['packet_size_variation'] = data[['src2dst_min_ps', 'src2dst_max_ps', 'dst2src_min_ps', 'dst2src_max_ps']].std(axis=1)
    
    data['is_udp_request'] = (data['protocol'] == 17) #Filter  UDP packets
    data['is_tcp_request'] = data['protocol'] == 6 # Filter only TCP requests
    data['is_icmp_request'] = (data['protocol'] == 1)# Filter ICMP

    
    # Calculate rolling window count of UDP requests
    data['Rolling_UDP_Requests_SourceDestination'] = data.groupby('src_dst_encoded')['is_udp_request'].apply(lambda x: udp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_UDP_Requests_Destination'] = data.groupby('dst_ip_encoded')['is_udp_request'].apply(lambda x: udp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Calculate rolling window count of TCP requests by source-destination pair
    data['Rolling_TCP_Requests_SourceDestination'] = data.groupby('src_dst_encoded')['is_tcp_request'].apply(lambda x: tcp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Calculate rolling window count of TCP requests by destination IP
    data['Rolling_TCP_Requests_Destination'] = data.groupby('dst_ip_encoded')['is_tcp_request'].apply(lambda x: tcp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Calculating Flags with respect to Src-Dst pair and dst reference
    data['Rolling_ACK_Packets_SourceDestination'] = data.groupby('src_dst_encoded')['bidirectional_ack_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_ACK_Packets_Destination'] = data.groupby('dst_ip_encoded')['bidirectional_ack_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_FIN_Packets_SourceDestination'] = data.groupby('src_dst_encoded')['bidirectional_fin_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_FIN_Packets_Destination'] = data.groupby('dst_ip_encoded')['bidirectional_fin_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_rst_Packets_SourceDestination'] = data.groupby('src_dst_encoded')['bidirectional_rst_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_rst_Packets_Destination'] = data.groupby('dst_ip_encoded')['bidirectional_rst_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_psh_Packets_SourceDestination'] = data.groupby('src_dst_encoded')['bidirectional_psh_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_psh_Packets_Destination'] = data.groupby('dst_ip_encoded')['bidirectional_psh_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_SYN_Packets_SourceDestination'] = data.groupby('src_dst_encoded')['bidirectional_syn_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_SYN_Packets_Destination'] = data.groupby('dst_ip_encoded')['bidirectional_syn_packets'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Calculate rolling window unique destination ports wrt to src-dst pair
    data['Unique_Ports_In_SourceDestinationIP'] = data.groupby('src_dst_encoded')['dst_port'].apply(lambda x: unique_ports_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Calculate rolling window count of ICMP requests wrt src-dst pair and dst 
    data['Rolling_ICMP_Requests_SourceDestination'] = data.groupby('src_dst_encoded')['is_icmp_request'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_ICMP_Requests_Destination'] = data.groupby('dst_ip_encoded')['is_icmp_request'].apply(lambda x: syn_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Create a new column that indicates whether the destination port is a commonly vulnerable port
    data['is_vulnerable_port'] = data['dst_port'].isin(http_ports)
    # Calculate rolling window count of https ports packets
    data['Rolling_http_port_SourceDestination'] = data.groupby('src_dst_encoded')['is_vulnerable_port'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_http_port_Destination'] = data.groupby('dst_ip_encoded')['is_vulnerable_port'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Calculating Average birectional duration rolling window
    data['Rolling_Duration_Destination'] = data.groupby('dst_ip_encoded')['bidirectional_duration_ms'].apply(lambda x: dur_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_Duration_SourceDestination'] = data.groupby('src_dst_encoded')['bidirectional_duration_ms'].apply(lambda x: dur_packets_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Create a new column that indicates whether the destination port is a commonly vulnerable port
    data['is_vulnerable_port'] = data['dst_port'].isin(dns_ports)
    # Calculate rolling window count of DNS requests wrt src-dst pair and dst only
    data['Rolling_DNS_request_SourceDestination'] = data.groupby('src_dst_encoded')['is_vulnerable_port'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_DNS_request_Destination'] = data.groupby('dst_ip_encoded')['is_vulnerable_port'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    # Create a new column that indicates whether the destination port is a commonly vulnerable port
    
    
    data['is_vulnerable_port'] = data['src_port'].isin(dns_ports)
    # Calculate rolling window count of ICMP requests
    data['Rolling_DNS_request_SourceDestination2'] = data.groupby('src_dst_encoded')['is_vulnerable_port'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_DNS_request_Destination2'] = data.groupby('dst_ip_encoded')['is_vulnerable_port'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)

    # Create a new column that indicates whether the destination port is a commonly vulnerable port
    data['is_vulnerable_port'] = data['dst_port'].isin(vulnerable_ports)
    # Calculate rolling window count of vulnerable ports
    data['Rolling_vulnerable_port'] = data.groupby('src_dst_encoded')['is_vulnerable_port'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_packets_destination'] = data.groupby('dst_ip_encoded')['src2dst_packets'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)
    data['Rolling_bipackets_destination'] = data.groupby('dst_ip_encoded')['bidirectional_packets'].apply(lambda x: icmp_requests_in_window(x, window_size)).reset_index(level=0, drop=True)

    data.drop(['src_dst_ip','src_dst_encoded','dst_ip_encoded','is_udp_request','is_tcp_request','is_icmp_request','is_vulnerable_port'],axis=1,inplace=True)
    return data
    
            