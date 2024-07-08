from torch_geometric.nn import HeteroConv, Linear, SAGEConv, global_mean_pool, Sequential, to_hetero,GraphConv,GATConv
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch
import torch.nn as nn

import deepsnap

 

def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    convs = {}
    for message_type in hetero_graph.edge_types:
        if first_layer is True:
            src_type = message_type[0]
            dst_type = message_type[2]
            src_size = hetero_graph.num_node_features[src_type]
            dst_size = hetero_graph.num_node_features[dst_type]
            convs[message_type] = conv(src_size, dst_size, hidden_size)
        else:
            convs[message_type] = conv(hidden_size, hidden_size, hidden_size)    
    return convs
    
    
class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        self.lin_src = nn.Linear(self.in_channels_src, self.out_channels)
        self.lin_dst = nn.Linear(self.in_channels_dst, self.out_channels)
        self.lin_update = nn.Linear(self.out_channels, self.out_channels)

    def forward(
        self,
        node_feature_src,
        node_feature_dst,
        edge_index,
        size=None
    ):
        return self.propagate(
            edge_index, size=size,
            node_feature_dst=node_feature_dst,
            node_feature_src=node_feature_src
        )

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = matmul(edge_index, node_feature_src, reduce="mean")
        return out

    def update(self, aggr_out, node_feature_dst):
        aggr_out = self.lin_src(aggr_out)
        node_feature_dst = self.lin_dst(node_feature_dst)
        aggr_out = torch.cat([aggr_out, node_feature_dst], dim=0)
        aggr_out = self.lin_update(aggr_out)
        return aggr_out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr


        self.mapping = {}
        self.alpha = None

        if self.aggr == "attn":
            self.attn_proj = nn.Sequential(
                nn.Linear(args['hidden_size'], args['attn_size']),
                nn.Tanh(),
                nn.Linear(args['attn_size'], 1, bias=False)
            )
    
    def reset_parameters(self):
        super(HeteroConvWrapper, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()
    
    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            ## Transforming into Sparse Tensor
            edge_index = SparseTensor.from_edge_index(edge_index.to(torch.long))
            
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src.float(),
                    node_feature_dst.float(),
                    edge_index,
                )
            )
            
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}    
        
        for (src, edge_type, dst), item in message_type_emb.items():
            if src==dst:
                mapping[1] = (src, edge_type, dst)
                node_emb['packet'].append(item)
            else:
                mapping[0] = (src, edge_type, dst)
                node_emb['flow'].append(item)
                
        self.mapping = mapping
        
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb
    
    def aggregate(self, xs):
        if self.aggr == "mean":
            x = torch.stack(xs, dim=-1)
            return x.mean(dim=-1)
        elif self.aggr == "attn":
            N = xs[0].shape[0] 
            M = len(xs) 

            x = torch.cat(xs, dim=0).view(M, N, -1) 
            z = self.attn_proj(x).view(M, N) 
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0) 
            self.alpha = alpha.view(-1).data.cpu().numpy()
            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        self.convs1 = HeteroConv({edge_type: SAGEConv((-1, -1), 64) for edge_type in hetero_graph.metadata()[1]})
        self.convs2 = HeteroConv({edge_type: SAGEConv((-1, -1), 64) for edge_type in hetero_graph.metadata()[1]})

        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
#             self.post_mps[node_type] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(node_type))
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()
            
        self.graph_prediction = nn.Linear(128, 64) 
        self.graph_prediction_1 = nn.Linear(64, 16)
        self.graph_prediction_2 = nn.Linear(16, 8) ## Number of Classes

    def forward(self, node_feature, edge_index,batch):
        x = node_feature

        x = self.convs1(x, edge_index)
        
        x = {key: self.bns1[key](value) for key, value in x.items()}
        x = {key: self.relus1[key](value) for key, value in x.items()}
        
#         x = {key: F.leaky_relu(value) for key, value in x.items()}
        
        x = self.convs2(x, edge_index)
        
        x = {key: self.bns2[key](value) for key, value in x.items()}
        x = {key: self.relus2[key](value) for key, value in x.items()}
        
#         x = {key: F.leaky_relu(value) for key, value in x.items()}      

        graph_emb = {key: pyg_nn.global_mean_pool(x[key], batch.batch_dict[key]) for key in batch.node_types}
        graph_emb = torch.cat([graph_emb[t] for t in graph_emb.keys()], dim=1)
        
        
        graph_pred = self.graph_prediction(graph_emb)
        graph_pred = self.graph_prediction_1(graph_pred)
        graph_pred = self.graph_prediction_2(graph_pred)

        graph_pred = F.log_softmax(graph_pred, dim=1)
        
        return graph_pred

    def loss(self, preds, label):
        return F.nll_loss(preds, label)
        
        
class HeteroGNN_Edge(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN_Edge, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        self.convs1 = HeteroConv({edge_type: GATConv((-1, -1), 64, edge_dim=-1, add_self_loops=False) for edge_type in hetero_graph.metadata()[1]})
        self.convs2 = HeteroConv({edge_type: GATConv((-1, -1), 64, edge_dim=-1,add_self_loops=False) for edge_type in hetero_graph.metadata()[1]})


        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=args['eps'])
#             self.post_mps[node_type] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(node_type))
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()
            
        self.graph_prediction = nn.Linear(128, 64) 
        self.graph_prediction_1 = nn.Linear(64, 16)
        self.graph_prediction_2 = nn.Linear(16, 8) ## Number of Classes

    def forward(self, node_feature, edge_index, edge_attr, batch):
        x = node_feature

        x = self.convs1(x, edge_index, edge_attr)
        
        x = {key: self.bns1[key](value) for key, value in x.items()}
        x = {key: self.relus1[key](value) for key, value in x.items()}
        
#         x = {key: F.leaky_relu(value) for key, value in x.items()}
        
        x = self.convs2(x, edge_index, edge_attr)
        
        x = {key: self.bns2[key](value) for key, value in x.items()}
        x = {key: self.relus2[key](value) for key, value in x.items()}
        
#         x = {key: F.leaky_relu(value) for key, value in x.items()}      

        graph_emb = {key: pyg_nn.global_mean_pool(x[key], batch.batch_dict[key]) for key in batch.node_types}
        graph_emb = torch.cat([graph_emb[t] for t in graph_emb.keys()], dim=1)
        
        
        graph_pred = self.graph_prediction(graph_emb)
        graph_pred = self.graph_prediction_1(graph_pred)
        graph_pred = self.graph_prediction_2(graph_pred)

        graph_pred = F.log_softmax(graph_pred, dim=1)
        
        return graph_pred

    def loss(self, preds, label):
        return F.nll_loss(preds, label)