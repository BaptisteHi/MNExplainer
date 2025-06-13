import torch_scatter
from graphmuse.nn.models.metrical_gnn import MetricalGNN
import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import F1Score, Accuracy


class CadenceGNNPytorch(nn.Module):
    def __init__(self, metadata, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5, hybrid=False, use_pitch_spelling=True):
        super(CadenceGNNPytorch, self).__init__()
        self.gnn = MetricalGNN(
            input_channels=input_dim, hidden_channels=hidden_dim, output_channels=hidden_dim // 2,
            num_layers=num_layers, metadata=metadata, dropout=dropout, fast=True)
        if use_pitch_spelling:
            self.pitch_spelling_emb = nn.Sequential(
                nn.Embedding(38, input_dim),
                nn.LayerNorm(input_dim))
            self.use_pitch_spelling = use_pitch_spelling
            self.input_projection = nn.Sequential(
                nn.Linear(2*input_dim, input_dim),
                nn.ReLU(),
                nn.LayerNorm(input_dim)
            )
        hidden_dim = hidden_dim // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.hybrid = hybrid
        if self.hybrid:
            self.rnn = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim//2, num_layers=2, batch_first=True, bidirectional=True,
                dropout=dropout)
            self.rnn_norm = nn.LayerNorm(hidden_dim)
            self.rnn_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.cat_proj = nn.Linear(hidden_dim*2, hidden_dim)
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cad_clf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def hybrid_forward(self, x, batch):
        lengths = torch.bincount(batch)
        x = x.split(lengths.tolist())
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
        x, _ = self.rnn(x)
        x = self.rnn_norm(x)
        x = self.rnn_mlp(x)
        x = nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=lengths)
        x = torch.cat(x, dim=0)
        return x

    def encode(self, x_dict, edge_index_dict, batch_size, neighbor_mask_node, neighbor_mask_edge, batch_dict=None, pitch_spelling=None):
        if batch_dict is None:
            batch_note = torch.zeros((x_dict["note"].shape[0], ), device=x_dict["note"].device).long()
        else:
            batch_note = batch_dict["note"]
        onset_mask = torch.zeros_like(batch_note).bool()
        onset_mask[:batch_size] = True

        if self.use_pitch_spelling:
            x_dict["note"] =  self.input_projection(torch.cat((x_dict["note"], self.pitch_spelling_emb(pitch_spelling)), dim=-1))

        x = self.gnn(
            x_dict, edge_index_dict, neighbor_mask_node=neighbor_mask_node, neighbor_mask_edge=neighbor_mask_edge)
        x = x[:batch_size]
        if self.hybrid:
            z = self.hybrid_forward(x_dict["note"][:batch_size], batch_note[:batch_size])
            x = self.cat_proj(torch.cat((x, z), dim=-1))

        onset_index = edge_index_dict["note", "onset", "note"]
        # remove self loops
        onset_index = onset_index[:, onset_index[0] != onset_index[1]]
        # apply onset_mask
        onset_index = onset_index[:, onset_mask[onset_index[0]] & onset_mask[onset_index[1]]]
        x = torch_scatter.scatter_mean(x[onset_index[0]], onset_index[1], dim=0, out=x.clone())
        x = self.norm(x)
        x = self.pool_mlp(x)
        return x

    def forward(self, x_dict, edge_index_dict, batch_size, neighbor_mask_node=None, neighbor_mask_edge=None, batch_dict=None):
        if neighbor_mask_node is None:
            neighbor_mask_node = {k: torch.zeros((x_dict[k].shape[0], ), device=x_dict[k].device).long() for k in x_dict}
        if neighbor_mask_edge is None:
            neighbor_mask_edge = {k: torch.zeros((edge_index_dict[k].shape[-1], ), device=edge_index_dict[k].device).long() for k in edge_index_dict}
        x = self.encode(x_dict, edge_index_dict, batch_size, neighbor_mask_node, neighbor_mask_edge, batch_dict)
        logits = self.cad_clf(x)
        return torch.softmax(logits, dim=-1)

    def clf(self, x):
        return self.cad_clf(x)



class CadencePLModel(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            metadata,
            dropout=0.5,
            lr=0.0001,
            weight_decay=5e-4,
            subgraph_size=500,
            reg_loss_weight=0.1,
            hybrid=False
    ):
        super(CadencePLModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.subgraph_size = subgraph_size
        self.reg_loss_weight = reg_loss_weight
        self.module = CadenceGNNPytorch(
            metadata=metadata, input_dim=input_dim, hidden_dim=hidden_dim,
            output_dim=output_dim, num_layers=num_layers, dropout=dropout, hybrid=hybrid)
        self.smote = SMOTE(dims=hidden_dim//2, distance='cosine', k=5)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.f1 = F1Score(num_classes=output_dim, task="multiclass", average="macro")
        self.acc = Accuracy(task="multiclass", num_classes=output_dim)

    def _common_step(self, batch, batch_idx):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        # out = model(batch.x_dict, batch.edge_index_dict)
        neighbor_mask_node = batch.num_sampled_nodes_dict
        neighbor_mask_edge = batch.num_sampled_edges_dict
        batch_size = batch["note"].batch_size
        batch_dict = batch.batch_dict
        pitch_spelling = batch["note"].pitch_spelling
        x = self.module.encode(
            x_dict, edge_index_dict, batch_size,
            neighbor_mask_node=neighbor_mask_node, neighbor_mask_edge=neighbor_mask_edge,
            batch_dict=batch_dict, pitch_spelling=pitch_spelling
        )
        # Trim the labels to the target nodes (i.e. layer 0)
        y = batch["note"].y[:batch_size]
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._common_step(batch, batch_idx)
        # feature loss verifies that the features are not too different
        feature_loss = x.pow(2).mean()
        x_over, y_over = self.smote.fit_generate(x, y)
        # Penalize when distance is too large between original and synthetic samples of the same class
        # Calculate Euclidean distance between synthetic and original samples
        for class_label in y_over.unique():
            mask = y_over == class_label
            x_over_class = x_over[mask]
            x_class = x[y == class_label]
            # Sample a few points from x_class and x_over to reduce computational cost
            if len(x_class) > 100:
                indices = np.random.choice(len(x_class), 100, replace=False)
                x_class = x_class[indices]
            if len(x_over_class) > 100:
                indices = np.random.choice(len(x_over_class), 100, replace=False)
                x_over_class = x_over_class[indices]
            distances = torch.cdist(x_over_class, x_class)
            min_distances, _ = torch.min(distances, dim=1)
            # Add penalty if distance is too large
            threshold = 1.0  # Set your own threshold
            penalties = torch.clamp(min_distances - threshold, min=0)
            feature_loss += penalties.mean()

        logits = self.module.clf(x_over)
        loss = self.loss(logits, y_over.long()) + (self.reg_loss_weight * feature_loss * (self.current_epoch*0.01))
        self.log('train_loss', loss.item(), batch_size=len(y), prog_bar=True)
        self.log('train_f1', self.f1(logits, y_over.long()), prog_bar=True, batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._common_step(batch, batch_idx)
        logits = self.module.cad_clf(x)
        num_classes = logits.shape[-1]
        # make the loss weighted by the number of labels
        num_labels = torch.bincount(y)
        # fix the shape of num_labels to match the shape of logits
        num_labels = torch.cat([num_labels, torch.zeros(num_classes - num_labels.shape[0], device=num_labels.device)])
        weight = 1 / (num_labels.float() + 1e-6) # avoid division by zero
        loss = F.cross_entropy(logits, y.long(), weight=weight)
        self.log('val_loss', loss.item(), batch_size=len(y), prog_bar=True)
        self.log('val_acc', self.acc(logits, y.long()), prog_bar=True, batch_size=len(y))
        self.log('val_f1', self.f1(logits, y.long()), prog_bar=True, batch_size=len(y))

    def test_step(self, batch, batch_idx):
        x, y = self._common_step(batch, batch_idx)
        logits = self.module.cad_clf(x)
        loss = self.loss(logits, y.long())
        self.log('test_loss', loss.item(), batch_size=len(y), prog_bar=True)
        self.log('test_acc', self.acc(logits, y.long()), prog_bar=True, batch_size=len(y))
        self.log('test_f1', self.f1(logits, y.long()), prog_bar=True, batch_size=len(y))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 40, 80], gamma=0.2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}