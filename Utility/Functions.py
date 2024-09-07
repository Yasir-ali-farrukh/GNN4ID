import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import os
import re
import glob
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
import sys
import random


class NIDSDataset(Dataset):

    """A dataset classfor  generates graph data objects from a provided
    CSV File/Files. Build uopn torch_geometric.data.Dataset base class for 
    creating graph datasets.

    Args:
        root (str): Root directory where the dataset should be saved.
        
        label_dict (Dict): Dictionary for assigning labels to each attack class.

        filename(List[str]): List of CSV file paths to be used for the development of 
            graph objects.

        skip_processing (bool): If set to `True`, skips the generation of graph 
            objects and utilizes the ones present in the root directory. 
            (default: `False`)

        test (bool): If set to `True`, generates data objects for testing by 
            creating data objects with a test suffix. 
            (default: `False`)

        index_num (int): Index to start with saving the grapgh object. Useful 
            when creating continuing grapgh object and you have to run the same
            class twice.
            (default: 0)

        include_packetflag (bool): If set to True, will also include Flags status
            as packet node features.
            (default: False)

        include_packetpayload (bool): If set to False, will not include payload 
            information as a packet node features.
            (default: True) 

        single_file (bool): If set to True, the provided CSV files is a single 
            file with Label column within CSV.
            (default: False)      

        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)

        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
    """
    
    def __init__(self, root, label_dict, filename, index_num = 0, skip_processing=False, include_packetflag=False, include_packetpayload=True, test=False, single_file=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.include_packetflag = include_packetflag
        self.include_packetpayload = include_packetpayload
        self.index = index_num
        self.label_dict = label_dict
        self.skip_processing = skip_processing
        self.length = 0
        self.single_file = single_file
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
                list_process = glob.glob(os.path.join(self.root, 'processed/data_test*'))
                self.length = len(list_process)
                return list_process[0]
            else:
                list_process = glob.glob(os.path.join(self.root, 'processed/data*'))
                self.length = len(list_process) - len(glob.glob(os.path.join(self.root, 'processed/data_test*')))
                return list_process[0]
        else:
            return []

    def download(self):
        pass

    def process(self):

        for files in self.raw_paths:
            
            self.data = pd.read_csv(files)
            print('Reading File ---> '+os.path.basename(files), file=sys.stderr)

            if not self.single_file:
                # Removing Some features that might seems unneccesssary. Features can be added to the node features by removing from the drop list.
                self.data.drop(['src_ip','src_port','dst_ip','dst_port','ip_version'], axis=1, inplace=True)
                self.data.drop(['bidirectional_bytes','bidirectional_first_seen_ms','bidirectional_last_seen_ms','bidirectional_duration_ms',
                         'bidirectional_packets','src2dst_first_seen_ms','src2dst_last_seen_ms','dst2src_first_seen_ms','dst2src_last_seen_ms',
                         'id','src_mac','src_oui','dst_mac','dst_oui','vlan_id','tunnel_id','bidirectional_syn_packets','bidirectional_cwr_packets',
                         'bidirectional_ece_packets','bidirectional_urg_packets','bidirectional_ack_packets','bidirectional_psh_packets',
                         'bidirectional_rst_packets','bidirectional_fin_packets'], axis=1, inplace=True)
                
                # Creating Dummy variables for Expiration_ID and protocol
                self.data['expiration_id']=pd.Categorical(self.data['expiration_id'], categories=[0,-1])
                # Creating Dummy varaible for protocol, make sure to incorporate all the protocols. There are only 5 protocols in the CIC-IoT2023 dataset. Add protocol number if utilizing other dataset.
                self.data['protocol']=pd.Categorical(self.data['protocol'], categories=[1,2,6,17,58])
                self.data=pd.get_dummies(self.data, prefix=['Exp','proto'], columns=['expiration_id', 'protocol'],dtype=int)
                # Getting the Label from the file name and provided dictionary
                label = self._get_labels(files)

    
            ## Converting String into iterable list; needed for extracting individual packet features
            self.data['udps.payload_data'] = self.data['udps.payload_data'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.packet_direction'] = self.data['udps.packet_direction'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.ip_size'] = self.data['udps.ip_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.transport_size'] = self.data['udps.transport_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.payload_size'] = self.data['udps.payload_size'].map(lambda x: x.strip('][').replace("'","").split(', '))
            self.data['udps.delta_time'] = self.data['udps.delta_time'].map(lambda x: x.strip('][').replace("'","").split(', '))
    
            ## If include_packetflag set True, then 
            if self.include_packetflag==True:
                self.data['udps.syn'] = self.data['udps.syn'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.cwr'] = self.data['udps.cwr'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.ece'] = self.data['udps.ece'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.urg'] = self.data['udps.urg'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.ack'] = self.data['udps.ack'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.psh'] = self.data['udps.psh'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.rst'] = self.data['udps.rst'].map(lambda x: x.strip('][').replace("'","").split(', '))
                self.data['udps.fin'] = self.data['udps.fin'].map(lambda x: x.strip('][').replace("'","").split(', '))
            
            
            
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
                if self.single_file:
                    label = torch.tensor(np.asarray(flow["Label"]), dtype=torch.int64)
    
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
            
            if not self.skip_processing:
                if self.test:
                    list_process = glob.glob(os.path.join(self.root, 'processed/data_test*'))
                    self.length = len(list_process)
                else:
                    list_process = glob.glob(os.path.join(self.root, 'processed/data*'))
                    self.length = len(list_process) - len(glob.glob(os.path.join(self.root, 'processed/data_test*')))
                

    def _get_flow_node_features(self, flow):
        """ Extract the features for Flow_node
        This will return a 2d array of the shape
        [1 , Node Feature size] """
        
        ## Removing the columns that are not part of flow_node features
        if self.single_file:
            ## If Using a Single CSV File that have Label Column in it
            flow_data=flow.drop(['Label','udps.payload_data','udps.delta_time', 'udps.packet_direction', 'udps.ip_size','udps.transport_size', 
                    'udps.payload_size', 'udps.syn', 'udps.cwr','udps.ece', 'udps.urg', 'udps.ack', 'udps.psh', 'udps.rst', 'udps.fin'])
        else:
            ## If Using multiple CSV files that do not have Label Column in them
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
        all_packet_node_feats = torch.tensor(all_packet_node_feats, dtype=torch.float32)

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
        contain_edge_all_feats=torch.tensor(contain_edge_all_feats, dtype=torch.float32)

        return contain_edge_all_feats

    def _get_link_edge_features(self, flow):
        """ Extract the features for link_edge.
        Features for each link_edge is delta_time between the two packets.
        This will return a 2d array of the shape
        [Number of link_edges, link_edge Feature size]
        """
        link_edge_feats = np.asarray(flow['udps.delta_time'][1:], dtype=int)
        link_edge_feats=torch.tensor(link_edge_feats, dtype=torch.float32)
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
        contain_edge = torch.tensor(contain_edge, dtype=torch.int64)
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
        link_edge = torch.tensor(link_edge, dtype=torch.int64)
        return link_edge

    def _get_labels(self, file_name):

        name=os.path.basename(file_name)
        label=self.label_dict[name.split('-')[0]]
        return torch.tensor(np.asarray([label]), dtype=torch.int64)

    def len(self):

        return self.length
        
        

    def get(self, idx):
        """ _ Equivalent to __getitem__ in pytorch"""
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
        
        

def rename_files(directory, name_mapping):
    """
    Renames files in the specified directory based on a given name mapping.

    Args:
        directory (str): The path to the directory containing the files to be renamed.
        name_mapping (dict): A dictionary mapping old name parts to new names.

    Returns:
        None
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    # List all files in the directory
    directory = directory+"\**\*pcap"
    files = glob.glob(directory)

    # Iterate over each file
    for filename in files:
        # Check if the file is a regular file
        if os.path.isfile(filename):
            # Getting the Name of the Attack type
            old_name_search = filename.split('\\')[-1]
            old_name_search = old_name_search.split('.')[-2]
            old_name_search = re.sub(r'\d+', '', old_name_search)
            old_name_search = old_name_search[:-1] if old_name_search.endswith("_") else old_name_search
            # Mapping the name as per the provided dictionary
            new_name = name_mapping.get(old_name_search, old_name_search)
            # Combining new name with the directory for renaming 
            new_name = os.path.dirname(filename)+"\\"+new_name
            # Extracting the file number from the name
        try:
            number = extract_number(os.path.basename(filename))
            new_name=new_name+"_"+number+".pcap"
        except:
            new_name=new_name+"_"+"0.pcap" 
        # Rename the file
        os.rename(filename, new_name)
        print(f"Renamed: {os.path.basename(filename)} -> {os.path.basename(new_name)}")


def extract_number(file_name: str) -> str:
    """
    Extracts the first sequence of digits found in a file name.

    Args:
        file_name (str): The file name from which to extract the number.

    Returns:
        str: The first sequence of digits found in the file name.

    """
    return re.search(r'\d+', file_name).group(0)


def duplicate_rows(df, target_rows):
    """
    Duplicates the rows of a DataFrame to reach the target number of rows.

    Args:
        df (pd.DataFrame): The input DataFrame to be duplicated.
        target_rows (int): The desired number of rows in the output DataFrame.

    Returns:
        pd.DataFrame: The resulting DataFrame with the target number of rows.
    """
    current_rows = len(df)
    original_df = df.copy()
    itr = int(target_rows/df.shape[0])-1
    for i in range(itr):
        df = pd.concat([df, original_df], ignore_index=True)
    target_over = target_rows-(df.shape[0])
    df = random_pick_rows(df, original_df, target_over)
    return df

def random_pick_rows(df, original, over):
    """
    Randomly picks rows from the original DataFrame to reach the target number of rows.

    Args:
        df (pd.DataFrame): The DataFrame to which rows will be added.
        original (pd.DataFrame): The original DataFrame from which rows will be picked.
        over (int): The number of additional rows needed to reach the target.

    Returns:
        pd.DataFrame: The resulting DataFrame with the additional rows.
    """
    for i in range(over):
        row_to_duplicate = random.randint(0, original.shape[0]-1)
        df = pd.concat([df, original.iloc[row_to_duplicate:row_to_duplicate+1]], ignore_index=True)
    return df



def Combining_classes(directory,classes_list,Number_in_individaul_class=20000, Number_of_test_samples=4000, label_dict= {'Benign': 0,'WebBased': 1,'Spoofing': 2,'Recon': 3,'Mirai': 4,'Dos' : 5,'DDos': 6,'BruteForce': 7}):
    
    """
    Combines CSV files from specified classes into a single DataFrame, 
    ensuring a consistent number of samples for each class, and creates training and test datasets.

    Args:
        directory (str): The path to the directory containing the CSV files.
        classes_list (list): List of classes to combine.
        Number_in_individual_class (int): Target number of train samples per class.
        Number_of_test_samples (int): Target number of test samples per class.
        label_dict (dict): Dictionary mapping class names to labels.

    Returns:
        None: The function saves the training and test datasets as CSV files.
    """
    for each_class in tqdm(classes_list):
        df_list = []
        List_of_CSV_File =glob.glob(directory+each_class+"*")
        # Load each file and append the dataframe to the list
        for file in List_of_CSV_File:
            df = pd.read_csv(file)
            name_file = file.split('\\')[-1]
            name_file = name_file.split('.')[-2]
            name_file = name_file.split('-')[0]
            df_list.append(df)
        
        final_df = pd.concat(df_list, ignore_index=True)
        
        if final_df.shape[0]<=Number_in_individaul_class:
            final_df = duplicate_rows(final_df,Number_in_individaul_class)
        else:
            fraction = Number_in_individaul_class/final_df.shape[0]
            final_df = final_df.sample(frac=fraction)    
        
        final_df['Label'] = label_dict[name_file]
        # Directory for saving
        train_file_path = os.path.dirname(file)+'\\train\\'+name_file+'_train.csv'
        # Creating the Directory
        if not os.path.exists(os.path.dirname(train_file_path)): 
            os.makedirs(os.path.dirname(train_file_path))
        
        final_df.to_csv(train_file_path, index=False)

        ## For Test Data
        List_of_CSV_File =glob.glob(directory+"test\\"+each_class+"*")
        df_list = []
        for file in List_of_CSV_File:
            df = pd.read_csv(file)
            name_file = file.split('\\')[-1]
            name_file = name_file.split('.')[-2]
            name_file = name_file.split('-')[0]
            df_list.append(df)
            os.remove(file) 

        final_df = pd.concat(df_list, ignore_index=True)
        final_df['Label'] = label_dict[name_file]
        if final_df.shape[0]>Number_of_test_samples:
            final_df = final_df.sample(n = Number_of_test_samples, random_state=42)
        test_file_path = os.path.dirname(file)+'\\'+name_file+'_test.csv'
        final_df.to_csv(test_file_path, index=False)

def split_csv(file_path, test_sample = 4000 , Number_in_individaul_class=20000):
    """
    Splits a CSV file into training and test datasets based on specific criteria for benign and attack traffic.

    Args:
        file_path (str): The path to the CSV file.
        test_sample (int): Number of test samples to extract.
        Number_in_individual_class (int): Number of training samples per class.

    Returns:
        None: The function saves the training and test datasets as CSV files.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    name_file = file_path.split('\\')[-1]
    name_file = name_file.split('.')[-2]
    # Name for filtering the Mac-addresses
    name_check = name_file.split('-')[0]
    
    if name_check == 'Benign':
        df[(df['src_mac']!='dc:a6:32:dc:27:d5') & (df['src_mac']!='e4:5f:01:55:90:c4') & (df['src_mac']!='dc:a6:32:c9:e4:ab') & (df['src_mac']!='ac:17:02:05:34:27') & (df['src_mac']!='dc:a6:32:c9:e5:a4') & (df['src_mac']!='dc:a6:32:c9:e4:d5') & (df['src_mac']!='dc:a6:32:c9:e5:ef') & (df['src_mac']!='dc:a6:32:c9:e4:90') & (df['src_mac']!='b0:09:da:3e:82:6c') & (df['dst_mac']!='dc:a6:32:dc:27:d5') & (df['dst_mac']!='e4:5f:01:55:90:c4') & (df['dst_mac']!='dc:a6:32:c9:e4:ab') & (df['dst_mac']!='ac:17:02:05:34:27') & (df['dst_mac']!='dc:a6:32:c9:e5:a4') & (df['dst_mac']!='dc:a6:32:c9:e4:d5') & (df['dst_mac']!='dc:a6:32:c9:e5:ef') & (df['dst_mac']!='dc:a6:32:c9:e4:90') & (df['dst_mac']!='b0:09:da:3e:82:6c')]
    else:
        df=df[(df['src_mac']=='dc:a6:32:dc:27:d5') | (df['src_mac']=='e4:5f:01:55:90:c4') | (df['src_mac']=='dc:a6:32:c9:e4:ab') | (df['src_mac']=='ac:17:02:05:34:27') | (df['src_mac']=='dc:a6:32:c9:e5:a4') | (df['src_mac']=='dc:a6:32:c9:e4:d5') | (df['src_mac']=='dc:a6:32:c9:e5:ef') | (df['src_mac']=='dc:a6:32:c9:e4:90') | (df['src_mac']=='b0:09:da:3e:82:6c') | (df['dst_mac']=='dc:a6:32:dc:27:d5') | (df['dst_mac']=='e4:5f:01:55:90:c4') | (df['dst_mac']=='dc:a6:32:c9:e4:ab') | (df['dst_mac']=='ac:17:02:05:34:27') | (df['dst_mac']=='dc:a6:32:c9:e5:a4') | (df['dst_mac']=='dc:a6:32:c9:e4:d5') | (df['dst_mac']=='dc:a6:32:c9:e5:ef') | (df['dst_mac']=='dc:a6:32:c9:e4:90') | (df['dst_mac']=='b0:09:da:3e:82:6c')]
    
    if df.shape[0] > 35000:
        df_test = df.sample(n = test_sample, random_state=42)
        df = df.drop(df_test.index)
        df = df.sample(n = Number_in_individaul_class, random_state=42)
    else:
        df_test = df.sample(frac=0.2, random_state=42)
        df = df.drop(df_test.index)
        
    test_file_path = os.path.dirname(file_path)+'\\test\\'+name_file+'_test.csv'
    
    if not os.path.exists(os.path.dirname(test_file_path)): # Creating the Directory
        os.makedirs(os.path.dirname(test_file_path))
        
    df_test.to_csv(test_file_path, index=False)
    df.to_csv(file_path, index=False)