import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch_geometric.utils as utils

train_data = pd.read_csv('training_songs_gnn.csv')

train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)

def preprocess_data(train_df, val_df):
    all_data = pd.concat([train_df, val_df], ignore_index=True)

    unique_songs = all_data['track_id'].unique()
    song_to_idx = {song: idx for idx, song in enumerate(unique_songs)}
    num_nodes = len(unique_songs)  # Total number of unique nodes

    x = torch.eye(num_nodes, dtype=torch.float)

    edges = []
    edge_attrs = []
    for _, row in all_data.iterrows():
        if pd.notna(row['similars']) and isinstance(row['similars'], str):
            src = song_to_idx[row['track_id']]
            dests = [song_to_idx[dest] for dest in row['similars'].split(',')]
            scores = [float(score) for score in row['sim_scores'].split(',')]
            edges.extend([(src, dest) for dest in dests])
            edge_attrs.extend(scores)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    degree = utils.degree(edge_index[0], num_nodes)
    x = degree.view(-1, 1).float()

    train_idx = torch.tensor([song_to_idx[song] for song in train_df['track_id']], dtype=torch.long)
    val_idx = torch.tensor([song_to_idx[song] for song in val_df['track_id']], dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes,
                train_mask=train_mask, val_mask=val_mask)

    return data


class SongRecommendationDataset(InMemoryDataset):
    def __init__(self, root, data, transform=None, pre_transform=None):
        super(SongRecommendationDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate([data])

    def _download(self):
        pass

    def _process(self):
        pass


class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, apply_pooling=False):
        super(GNNModel, self).__init__()
        self.apply_pooling = apply_pooling
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.8)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.8)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.pool3 = TopKPooling(hidden_channels, ratio=0.8)
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        if self.apply_pooling:
            x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        if self.apply_pooling:
            x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        if self.apply_pooling:
            x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)

        return x



data = preprocess_data(train_df, val_df)
dataset = SongRecommendationDataset(root='.', data=data)

# Split data into train and validation loaders
train_loader = dataset.data.train_mask.nonzero().view(-1)
val_loader = dataset.data.val_mask.nonzero().view(-1)

# Train the model
model = GNNModel(in_channels=1, hidden_channels=64, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    node_repr = model(data.x, data.edge_index, data.edge_attr)
    out = node_repr[data.train_mask]
    loss = criterion(out, torch.ones_like(out)) 
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        node_repr = model(data.x, data.edge_index, data.edge_attr)
        val_out = node_repr[data.val_mask]
        val_loss = criterion(val_out, torch.ones_like(val_out))
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')