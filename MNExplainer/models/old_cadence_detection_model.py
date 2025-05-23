import torch
import torch_scatter
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv, MessagePassing, HGTConv, JumpingKnowledge
from graphmuse.utils.graph_utils import trim_to_layer

class SMOTE(object):
    """
    Minority Sampling with SMOTE.
    """
    def __init__(self, distance='custom', dims=512, k=2):
        super(SMOTE, self).__init__()
        self.newindex = 0
        self.k = k
        self.dims = dims
        self.distance_measure = distance

    def populate(self, N, i, nnarray, min_samples, k, device='cpu'):
        new_index = torch.arange(self.newindex, self.newindex + N, dtype=torch.int64, device=device)
        nn = torch.randint(0, k-1, (N, ), dtype=torch.int64, device=device)
        diff = min_samples[nnarray[nn]] - min_samples[i]
        gap = torch.rand((N, self.dims), dtype=torch.float32, device=device)
        self.synthetic_arr[new_index] = min_samples[i] + gap * diff
        self.newindex += N

    def k_neighbors(self, euclid_distance, k, device='cpu'):
        nearest_idx = torch.zeros((euclid_distance.shape[0], euclid_distance.shape[0]), dtype=torch.int64, device=device)

        idxs = torch.argsort(euclid_distance, dim=1)
        nearest_idx[:, :] = idxs

        return nearest_idx[:, 1:k+1]

    def find_k(self, X, k, device='cpu'):
        z = F.normalize(X, p=2, dim=1)
        distance = torch.mm(z, z.t())
        return self.k_neighbors(distance, k, device=device)

    def find_k_euc(self, X, k, device='cpu'):
        euclid_distance = torch.cdist(X, X)
        return self.k_neighbors(euclid_distance, k, device=device)

    def find_k_cos(self, X, k, device='cpu'):
        cosine_distance = F.cosine_similarity(X, X)
        return self.k_neighbors(cosine_distance, k, device=device)

    def generate(self, min_samples, N, k, device='cpu'):
        """
        Returns (N/100) * n_minority_samples synthetic minority samples.
        Parameters
        ----------
        min_samples : Numpy_array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples
        N : percetange of new synthetic samples:
            n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
        k : int. Number of nearest neighbours.
        Returns
        -------
        S : Synthetic samples. array,
            shape = [(N/100) * n_minority_samples, n_features].
        """
        T = min_samples.shape[0]
        self.synthetic_arr = torch.zeros(int(N / 100) * T, self.dims, dtype=torch.float32, device=device)
        N = int(N / 100)
        if self.distance_measure == 'euclidian':
            indices = self.find_k_euc(min_samples, k, device=device)
        elif self.distance_measure == 'cosine':
            indices = self.find_k_cos(min_samples, k, device=device)
        else:
            indices = self.find_k(min_samples, k, device=device)
        for i in range(indices.shape[0]):
            self.populate(N, i, indices[i], min_samples, k, device=device)
        self.newindex = 0
        return self.synthetic_arr

    def fit_generate(self, X, y):
        """
        Over-samples using SMOTE. Returns synthetic samples concatenated at the end of the original samples.
        Parameters
        ----------
        X : Numpy_array-like, shape = [n_samples, n_features]
            The input features
        y : Numpy_array-like, shape = [n_samples]
            The target labels.

        Returns
        -------
        X_resampled : Numpy_array, shape = [(n_samples + n_synthetic_samples), n_features]
            The array containing the original and synthetic samples.
        y_resampled : Numpy_array, shape = [(n_samples + n_synthetic_samples)]
            The corresponding labels of `X_resampled`.
        """
        # get occurence of each class
        occ = torch.eye(int(y.max() + 1), int(y.max() + 1), device=X.device)[y].sum(axis=0)
        # get the dominant class
        dominant_class = torch.argmax(occ)
        # get occurence of the dominant class
        n_occ = int(occ[dominant_class].item())
        for i in range(len(occ)):
            # For Mini-Batch Training exclude examples with less than k occurances in the mini banch.
            if i != dominant_class and occ[i] >= self.k:
                # calculate the amount of synthetic data to generate
                N = (n_occ - occ[i]) * 100 / occ[i]
                if N != 0:
                    candidates = X[y == i]
                    xs = self.generate(candidates, N, self.k, device=X.device)
                    X = torch.cat((X, xs))
                    ys = torch.ones(xs.shape[0], device=y.device) * i
                    y = torch.cat((y, ys))
        return X, y.long()


class EncodingGNN(nn.Module):

    def __init__(self, metadata, input_channels, hidden_channels, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.edge_types = metadata[1]
        self.convs.append(
            HeteroConv(
                {
                    edge_type: SAGEConv(input_channels, hidden_channels, normalize=True, project=True)
                    for edge_type in self.edge_types
                }, aggr='mean')
        )
        for _ in range(num_layers-1):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(hidden_channels, hidden_channels)
                    for edge_type in self.edge_types
                }, aggr='mean')
            self.convs.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_channels))

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge):

        for i, conv in enumerate(self.convs[:-1]):
            if not neighbor_mask_edge is None and not neighbor_mask_node is None:
                x_dict, edge_index_dict, _ = trim_to_layer(
                    layer=self.num_layers - i,
                    neighbor_mask_node=neighbor_mask_node,
                    neighbor_mask_edge=neighbor_mask_edge,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.layer_norms[i](x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        if not neighbor_mask_edge is None and not neighbor_mask_node is None:
            x_dict, edge_index_dict, _ = trim_to_layer(
                layer=1,
                neighbor_mask_node=neighbor_mask_node,
                neighbor_mask_edge=neighbor_mask_edge,
                x=x_dict,
                edge_index=edge_index_dict,
            )
        x_dict = self.convs[-1](x_dict, edge_index_dict)        
        return x_dict

class CadenceDetectionGNN(nn.Module):

    def __init__(self, metadata, input_channels, hidden_channels, num_layers, num_cad_type, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.encoder = EncodingGNN(metadata, input_channels, hidden_channels, num_layers, dropout)
        self.smote = SMOTE(dims=hidden_channels, k=3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, hidden_channels//2),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels//2),
            Linear(hidden_channels//2, num_cad_type + 1) #the +1 here is for the case where the node is not associated with any cadences.
        )
    
    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None, neighbor_mask_edge=None, ground_truth=None):

        #encoding
        x_dict = self.encoder(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)

        if not neighbor_mask_edge is None:
            edge_index_dict['note','onset','note'] = edge_index_dict['note','onset','note'][:,neighbor_mask_edge['note','onset','note']==0]

        if not neighbor_mask_node is None:
            node_mask_b = neighbor_mask_node["note"]==0
            edge_mask_b = torch.logical_and(node_mask_b[edge_index_dict['note','onset','note'][0]],node_mask_b[edge_index_dict['note','onset','note'][1]])
            edge_index_dict['note','onset','note'] = edge_index_dict['note','onset','note'][:,edge_mask_b]

            x_plus_empty = torch.cat((x_dict['note'],torch.zeros(1, self.hidden_channels, device=x_dict['note'].device)), dim=0)
            initial_num_notes = len(neighbor_mask_node['note'])
            after_trim_num_notes = len(x_dict['note'])
            features = torch.full((initial_num_notes,), after_trim_num_notes, dtype=torch.long).to(x_dict['note'].device)
            features[neighbor_mask_node['note']==0] = torch.arange(after_trim_num_notes, dtype=torch.long, device=features.device)

            #features_extending_mask = []
            #j = 0
            #for i in range(initial_num_notes):
            #    is_ghost = neighbor_mask_node['note'][i]
            #    if is_ghost:
            #        features_extending_mask.append(after_trim_num_notes) #index of the empty column in x_plus_empty
            #    else:
            #        features_extending_mask.append(j)
            #        j += 1
    
            x = x_plus_empty[features]
            x.to(features.device)

        else:
            x = x_dict['note']
            x.to(x_dict['note'].device)

        #aggregating on onset division
        #onsets = torch.unique(notes_onsets)
        #for onset in onsets:
        #    filter = notes_onsets == onset
        #    notes = filter.nonzero()
        #    notes = [notes[i,0] for i in range(len(notes))]
        #    aggregating_vector = torch.sum(torch.stack([x_dict["note"][note,:] for note in notes]), dim=0)
        #    aggregating_vector *= (1/len(notes))
        #    x_dict["note"][notes,:] = aggregating_vector
        
        
        h = torch_scatter.scatter(x[edge_index_dict['note','onset','note'][0]], 
                                  edge_index_dict['note','onset','note'][1],
                                  0,
                                  out=torch.zeros(x.shape, device=x.device),
                                  reduce='mean')
        h.to(x.device)
        
        if not neighbor_mask_node is None:
            h = h[neighbor_mask_node['note']==0]

        #applying SMOTE layer
        if not ground_truth is None:
            h, smote_ground_truth = self.smote.fit_generate(h, ground_truth)
            #applying classifier
            h = self.classifier(h)
            return h, smote_ground_truth
        
        else:
            #applying classifier
            h = self.classifier(h)
            return h