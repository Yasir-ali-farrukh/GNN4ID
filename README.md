# <img src="https://github.com/user-attachments/assets/d29a73cc-120b-48e5-8e32-b3155410f586" width="120" valign="middle" alt="Scapy" />&nbsp; GNN4ID

<p align="justify">GNN4ID is a tool designed to transform raw packet capture files (PCAP) of network traffic into structured graph-based datasets. This tool uniquely integrates both flow and packet-level information, providing a comprehensive view of network activity. Developed to facilitate research in Graph Neural Networks (GNNs) for Network Intrusion Detection Systems (NIDS), GNN4ID empowers users to seamlessly extract flow-level information along with its respective packet-level information, which is ultimately combined to form a graph-based dataset. The developed graph-based dataset utilizes both flow-level and packet-level information. </p>

## Usage
<p align="justify">
GNN4ID can be utilized to extract and create graph objects from any network traffic data. The primary requirement is that the network traffic is available in its raw pcap format and that the pcap files or individual packets are appropriately labeled. If the provided pcap files are not labeled but their flow-level information is labeled, you can use the <a href="https://github.com/Yasir-ali-farrukh/Payload-Byte">Payload-Byte</a> Tool to label the raw packets with respect to their corresponding flow.

<p align="justify">
The input files required for GNN4ID are labeled pcap files or labeled packet information. Alternatively, you can use the provided flow information along with their respective packet-level information. For ease of usage, we have provided three notebook files that can be used to extract and generate graph data objects:


1. [`GNN4ID.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/GNN4ID.ipynb): This notebook provides instructions on how to handle pcap files for flow and packet-level information extraction and how to transform the extracted flow and packet-level information into graph data objects. You can either create the graph data objects directly or use the extracted combined information of flow and packet for your own use case.
2. [`Data_preprocessing_CIC-IoT2023.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/Data_preprocessing_CIC-IoT2023.ipynb): This notebook details the preprocessing of the CIC-IoT2023 dataset, specifically addressing the issue of class imbalance and providing the dataset sample size for our example.
3. [`GNN4ID_Model.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/GNN4ID_Model.ipynb): This notebook provides details on training and testing a GNN model. We have incorporated two different models, one with edge attributes and another without edge attributes.
</p>



To start, you can begin with [`GNN4ID.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/GNN4ID.ipynb), as it provides comprehensive details on information extraction from pcap files and the creation of graph models. For ease of computation, you should follow the steps in [`Data_preprocessing_CIC-IoT2023.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/Data_preprocessing_CIC-IoT2023.ipynb) before creating the data objects (as highlighted in the notebook)/Or download the processed CIC-IoT2023 dataset directly from the [Link](https://drive.google.com/drive/folders/1FiZh87vvCZF3gX1Fnj9iTB4j74u-nuR6?usp=drive_link). Lastly, [`GNN4ID_Model.ipynb`](https://github.com/Army-Cyber-Institute/intelligent-and-self-sustaining-nids/blob/main/Project_2_GNN_ID/GNN4ID_Model.ipynb) provides a baseline on how to develop a model, train it, and test it.

## Graph Data Modeling
<p align="justify">
To model network traffic data into a graph data structure, we have utilized the relationships between packets and how combined packets are used to create flow information. By leveraging this connection, we can transform network traffic into graph objects. We adopted heterogeneous graph modeling as it allows us to model two distinct types of nodes with their own attributes. Specifically, we used flow information as one node type and packet information as another. Consequently, our graph comprises flow nodes connected with their respective packets.

<p align="justify">
For our experimentation and real-time detection, we set a limit on the maximum number of packets in a flow to 20. This means that if a flow contains 20 packets, the flow is terminated, and a new flow is computed. This approach ensures real-time detection, preventing flows from containing a large number of packets, which could lead to several minutes of delay.

<p align="justify">
As we have two types of nodes, we also have two different types of edges: link edges and contain edges. The link edges connect packet nodes to packet nodes, while the contain edges connect flow nodes to packet nodes. The attributes of each node and edge are as follows:

1. **Flow Node**: 82 Features (Statistical Flow Features)
2. **Packet Node**: 1500 Features (Payload Data Byte-wise)
3. **Contain Edge**: 4 Features (Information of different packet layer sizes)
4. **Link Edge**: 1 Feature (Time delta between each consecutive packet)

A pictorial representation of the graph object is provided below:


<p align="center">
  <img src="https://github.com/user-attachments/assets/f9b971d2-848e-49de-94d2-ced5ca047d95" width="400" height="300">
</p>

## Citation 
 If you are using our tool, kindly cite our paper  [paper](https://arxiv.org/abs/2408.16021) which outlines the details of the graph modeling and processing. 


 ```yaml
@article{GNN4ID,
  title={XG-NID: Dual-Modality Network Intrusion Detection using a Heterogeneous Graph Neural Network and Large Language Model},
  author={Farrukh, Yasir Ali and Wali, Syed and Khan, Irfan and Bastian, Nathaniel D},
  journal={arXiv preprint arXiv:2408.16021},
  year={2024}
}
```


