# <img src="https://github.com/Yasir-ali-farrukh/GNN4ID/assets/93033074/91ff144c-550a-47b3-bc0a-907a77470347" width="120" valign="middle" alt="Scapy" />&nbsp; GNN4ID

<p align="justify">GNN4ID is a tool designed to transform raw packet capture files (PCAP) of network traffic into structured graph-based datasets. This tool uniquely integrates both flow and packet-level information, providing a comprehensive view of network activity. Developed to facilitate research in Graph Neural Networks (GNNs) for Network Intrusion Detection Systems (NIDS), GNN4ID empowers users to seamlessly extract flow-level information along with its respective packet-level information, which is ultimately combined to form a graph-based dataset. The developed graph-based dataset utilizes both flow-level and packet-level information. </p>

## Usage
<p align="justify">
GNN4ID can be utilized to extract and create graph objects from any network traffic data. The primary requirement is that the network traffic is available in its raw pcap format and that the pcap files or individual packets are appropriately labeled. If the provided pcap files are not labeled but their flow-level information is labeled, you can use the <a href="https://github.com/Yasir-ali-farrukh/Payload-Byte">Payload-Byte</a> Tool to label the raw packets with respect to their corresponding flow.

<p align="justify">
The input files required for GNN4ID are labeled pcap files or labeled packet information. Alternatively, you can use the provided flow information along with their respective packet-level information. For ease of usage, we have provided three notebook files that can be used to extract and generate graph data objects:

1. `GNN4ID.ipynb`: This notebook provides instructions on how to handle pcap files for flow and packet-level information extraction and how to transform the extracted flow and packet-level information into graph data objects. You can either create the graph data objects directly or use the extracted combined information of flow and packet for your own use case.
2. `Data_preprocessing_CIC-IoT2023.ipynb`: This notebook details the preprocessing of the CIC-IoT2023 dataset, specifically addressing the issue of class imbalance and providing the dataset sample size for our example.
3. `GNN4ID_Heterogenous_Model.ipynb`: This notebook provides details on training and testing a GNN model. We have incorporated two different models, one with edge attributes and another without edge attributes.

To start, you can begin with `GNN4ID.ipynb`, as it provides comprehensive details on information extraction from pcap files and the creation of graph models. For ease of computation, you should follow the steps in `Data_preprocessing_CIC-IoT2023.ipynb` before creating the data objects (as highlighted in the notebook). Lastly, `GNN4ID_Heterogenous_Model.ipynb` provides a baseline on how to develop a model, train it, and test it.

</p>
## Requirement

