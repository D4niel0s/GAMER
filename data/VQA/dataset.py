import torch

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.data import Batch
from torch.utils.data import  Dataset

from model.modules.utils import vqa_answers_to_soft_label


class VQAGraphsDataset(Dataset):
    def __init__(self, hf_dataset, answer2idx, graph_builder=None, **kwargs):
        self.dataset = hf_dataset
        self.answer2idx = answer2idx
        self.graph_builder = graph_builder
        self.graph_building_args = kwargs
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        i_embed = item['image_embedding']
        q_embed = item['question_embedding']

        if self.graph_builder is not None:
            graph = self.graph_builder(
                text_embeds = q_embed.unsqueeze(0),
                image_embeds = i_embed.unsqueeze(0),
                attn_mask = torch.ones((1,q_embed.shape[0])),
                **self.graph_building_args
            )
        else:
            graph = None

        label = vqa_answers_to_soft_label(item['answers'], self.answer2idx)
        return graph, torch.tensor(label, dtype=torch.float32)


    @staticmethod
    def vqa_collate_fn(batch, add_lap_pe = True, lap_pe_transform = AddLaplacianEigenvectorPE(k=16, attr_name="lap_pe", is_undirected=True)):
        graphs, labels = zip(*batch)

        if add_lap_pe:
            graphs = [lap_pe_transform(graph) for graph in graphs]

        batch_graph = Batch.from_data_list(graphs)
        labels = torch.stack(labels, dim=0)

        return dict(
                    x = getattr(batch_graph, "x", None),                    # node features or None
                    edge_index = batch_graph.edge_index,                    # edge index
                    batch = batch_graph.batch,                              # graph id per node
                    edge_attr = getattr(batch_graph, "edge_attr", None),    # optional edge features
                    lap_pe = getattr(batch_graph, "lap_pe", None),          # optional LapPE
        ), labels