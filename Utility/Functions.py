import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import os
import glob
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset, Data
import sys


class NIDSDataset(Dataset):
    
    def __init__(self, root, label_dict, filename, index_num = 0, skip_processing=False, include_packetflag=False, include_packetpayload=True, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        ## List of files to be Transformed into Graph-Based-Neural-Network
        self.filename = filename
        self.include_packetflag = include_packetflag
        self.include_packetpayload = include_packetpayload
        self.index = index_num
        self.label_dict = label_dict
        self.skip_processing = skip_processing
        super(NIDSDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.skip_processing:
            if self.test:
                list_process = glob.glob(os.path.join(self.root, 'processed\data_test*'))
                return list_process[0]
            else:
                list_process = glob.glob(os.path.join(self.root, 'processed\data*'))
                return list_process[0]
        else:
            return []

    def download(self):
        pass

    def process(self):

        for files in self.raw_paths:
            
            self.data = pd.read_csv(files)
            print('Reading File ---> '+os.path.basename(files), file=sys.stderr)
    
            # Removing Some features that might seems unneccesssary. Features can be added to the node features by removing from the drop list.
            self.data.drop(['src_ip','src_port','dst_ip','dst_port','ip_version'], axis=1, inplace=True)
            self.data.drop(['bidirectional_bytes','bidirectional_first_seen_ms','bidirectional_last_seen_ms','bidirectional_duration_ms',
                     'bidirectional_packets','src2dst_first_seen_ms','src2dst_last_seen_ms','dst2src_first_seen_ms','dst2src_last_seen_ms',
                     'id','src_mac','src_oui','dst_mac','dst_oui','vlan_id','tunnel_id','bidirectional_syn_packets','bidirectional_cwr_packets',
                     'bidirectional_ece_packets','bidirectional_urg_packets','bidirectional_ack_packets','bidirectional_psh_packets',
                     'bidirectional_rst_packets','bidirectional_fin_packets'], axis=1, inplace=True)
            
            ## Creating Dummy variables for Expiration_ID and protocol
            self.data=pd.get_dummies(self.data, prefix=['Exp','proto'], columns=['expiration_id', 'protocol'],dtype=int)
    
            ## Converting String into iterable list; needed for extracting individual packet features
            self.data['udps.payload_data'] = self.data['udps.payload_data'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.packet_direction'] = self.data['udps.packet_direction'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.ip_size'] = self.data['udps.ip_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.transport_size'] = self.data['udps.transport_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.payload_size'] = self.data['udps.payload_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.delta_time'] = self.data['udps.delta_time'].map(lambda x: x.strip('][').replace("'","").split(', '))
    
            if self.include_packetflag==True:
                self.data['udps.syn'] = self.data['udps.syn'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.cwr'] = self.data['udps.cwr'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.ece'] = self.data['udps.ece'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.urg'] = self.data['udps.urg'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.ack'] = self.data['udps.ack'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.psh'] = self.data['udps.psh'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.rst'] = self.data['udps.rst'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.fin'] = self.data['udps.fin'].map(lambda x: x.strip('][').replace("'","").split(', '))
            
            label = self._get_labels(files)
            
            for index, flow in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                # Get Node Feaetures
                flow_node_feats = self._get_flow_node_features(flow) # Get Flow_node features
                packet_node_feats = self._get_packet_node_features(flow) # Get Packet_node features
                
                # Get Edge Index/Adjacency Matrix
                contain_edge_index = self._get_contain_edge_index(len(flow['udps.payload_data'])) # Get Contain_edge Index
                link_edge_index = self._get_link_edge_index(len(flow['udps.payload_data'])) # Get Link_edge Index
                
                # Get Edge Features/Attributes
                contain_edge_feats = self._get_contain_edge_features(flow) # Get contain_edge features
                link_edge_feats = self._get_link_edge_features(flow) # Get link_edge features
    
                ### Get labels info (If label is present in a dataset column. You may need to edit the _get_labels function as per the dataset columns.
                # label = self._get_labels(flow["label"]) ## 
    
                # Create data object
                data = HeteroData()
                data['flow'].x = flow_node_feats
                data['packet'].x = packet_node_feats
                data['flow', 'contain', 'packet'].edge_index = contain_edge_index
                data['packet', 'link', 'packet'].edge_index = link_edge_index
                data['flow', 'contain', 'packet'].edge_attr = contain_edge_feats
                data['packet', 'link', 'packet'].edge_attr = link_edge_feats
                data.y = label

                data = T.ToUndirected()(data)
    
                if self.test:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                     f'data_test_{self.index}.pt'))
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                     f'data_{self.index}.pt'))
                self.index+=1

    def _get_flow_node_features(self, flow):
        """ Extract the features for Flow_node
        This will return a 2d array of the shape
        [1 , Node Feature size] """
        
        ## Removing the columns that are not part of flow_node features
        flow_data=flow.drop(['udps.payload_data','udps.delta_time', 'udps.packet_direction', 'udps.ip_size','udps.transport_size', 
                   'udps.payload_size', 'udps.syn', 'udps.cwr','udps.ece', 'udps.urg', 'udps.ack', 'udps.psh', 'udps.rst', 'udps.fin'])
        
        ## Transforming into tensors for pytorch
        all_flow_node_feats = np.asarray(flow_data, dtype=float)
        all_flow_node_feats = np.reshape(all_flow_node_feats, (-1, all_flow_node_feats.shape[0]))
        all_flow_node_feats=torch.tensor(all_flow_node_feats, dtype=torch.float32)
        
        return all_flow_node_feats

    def _get_packet_node_features(self, flow):
        """ Extract the features for packet_node
        Features for each packet is its payload data. We can also incorporate flags data for each packet. 
        This will return a 2d array of the shape
        [Num_packets_in_flow, Node Feature size] """
        
        dims = 1500 ## Length for payload Bytes to be incorporated/Number for columns for payload data
        all_packet_node_feats=[]

        for index_value in range(len(flow['udps.payload_data'])):
            packet_feats_combined = []
            
            ## Each Packet Flags as Packet Node Features
            if self.include_packetflag==True:
                packet_feats_combined.append(flow['udps.syn'][index_value])
                packet_feats_combined.append(flow['udps.cwr'][index_value])
                packet_feats_combined.append(flow['udps.ece'][index_value])
                packet_feats_combined.append(flow['udps.urg'][index_value])
                packet_feats_combined.append(flow['udps.ack'][index_value])
                packet_feats_combined.append(flow['udps.psh'][index_value])
                packet_feats_combined.append(flow['udps.rst'][index_value])
                packet_feats_combined.append(flow['udps.fin'][index_value])

            ## Packet Payload as Packet Node Features
            if self.include_packetpayload==True:
                byte_array = bytes.fromhex(flow['udps.payload_data'][index_value])
                byte_lst = list(byte_array)
                if (len(byte_lst) < dims):
                    packet_feat = np.pad(byte_lst, (0, dims-len(byte_lst)), 'constant')
                else:
                    packet_feat = np.array(byte_lst[0:dims].copy())
                packet_feat = np.abs(np.uint8(packet_feat))
                packet_feats_combined.extend(packet_feat.tolist())

            all_packet_node_feats.append(packet_feats_combined)

        all_packet_node_feats = np.asarray(all_packet_node_feats, dtype=int)
        all_packet_node_feats = torch.tensor(all_packet_node_feats, dtype=torch.int)

        return all_packet_node_feats
            

    def _get_contain_edge_features(self, flow):
        """ Extract the features for contain_edge.
        Features for each contain_edge is packet_direction, ip_size, transport_size, payload_size of that particular packet.        
        This will return a 2d array of the shape
        [Number of contain_edges, contain_edge Feature size]
        """
        contain_edge_all_feats = []
        
        for index_value in range(len(flow['udps.packet_direction'])):
            contain_edge_feat = []
            contain_edge_feat.append(flow['udps.packet_direction'][index_value])
            contain_edge_feat.append(flow['udps.ip_size'][index_value])
            contain_edge_feat.append(flow['udps.transport_size'][index_value])
            contain_edge_feat.append(flow['udps.payload_size'][index_value])
            contain_edge_all_feats.append(contain_edge_feat)
    
        contain_edge_all_feats = np.asarray(contain_edge_all_feats, dtype=int)
        contain_edge_all_feats=torch.tensor(contain_edge_all_feats, dtype=torch.int)

        return contain_edge_all_feats

    def _get_link_edge_features(self, flow):
        """ Extract the features for link_edge.
        Features for each link_edge is delta_time between the two packets.
        This will return a 2d array of the shape
        [Number of link_edges, link_edge Feature size]
        """
        link_edge_feats = np.asarray(flow['udps.delta_time'][1:], dtype=int)
        link_edge_feats=torch.tensor(link_edge_feats, dtype=torch.int)
        return link_edge_feats

    def _get_contain_edge_index(self, length):
        """ Extract the adjacency matrix for the edges connection between flow_node and packet_nodes.
        Contain_Edge links the flow_node to each and every packet_node that it contains.
        For Example: If there are 20 packets in a particular flow then there will be 20 Contain_Edge.
                     As each packet is linked to the flow.
        This will return a matrix / 2d array of the shape
        [2, num_contain_edges]
        """
        Flow = np.zeros(length,dtype=int)
        Packet = np.arange(0, length)
        contain_edge = np.vstack ((Flow, Packet))
        contain_edge = torch.tensor(contain_edge, dtype=torch.int)
        return contain_edge

    def _get_link_edge_index(self, length):
        """ Extract the adjacency matrix for the edges connection between packet_nodes.
        Link_Edge links the packet_node to other packet_node available.
        For Example: Each packet is linked with its following packet. If there are 20 packets then 
                     there will be 19 link_edges.
        This will return a matrix / 2d array of the shape
        [2, num_link_edges]
        """
        packet_ini = np.arange(0, length-1)
        packet_next = np.arange(1, length)
        link_edge =  np.vstack ((packet_ini, packet_next))
        link_edge = torch.tensor(link_edge, dtype=torch.int)
        return link_edge

    def _get_labels(self, file_name):

        name=os.path.basename(file_name)
        label=self.label_dict[name.split('_')[0]]
        return torch.tensor(label, dtype=torch.int)

    def len(self):
        num_graphs = len(glob.glob(os.path.join(self.root, 'processed\data*')))
        num_graphs_test = len(glob.glob(os.path.join(self.root, 'processed\data_test*')))
        
        if self.test:
            return num_graphs_test
        else:
            return num_graphs-num_graphs_test

    def get(self, idx):
        """ _ Equivalent to __getitem__ in pytorch"""
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data