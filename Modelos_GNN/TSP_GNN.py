import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class RedNeuronalTSP(torch.nn.Module):
    """GNN basada en capas de atención (GAT) para predecir, por cada
    arista del grafo, la probabilidad de que pertenezca a la ruta óptima."""
    def __init__(self, dim_nodos=2, dim_oculta=128, num_capas=2):
        super(RedNeuronalTSP, self).__init__()
        self.proyeccion_inicial = nn.Linear(dim_nodos, dim_oculta)
        
        self.capas_gat = nn.ModuleList()
        self.norm_capas = nn.ModuleList()
        for _ in range(num_capas):
            self.capas_gat.append(GATConv(in_channels=dim_oculta, out_channels=dim_oculta, edge_dim=1))
            self.norm_capas.append(nn.LayerNorm(dim_oculta))

        self.clasificador_arista = nn.Sequential(
            nn.Linear(dim_oculta * 2, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, 1)
        )

    def forward(self, datos):
        x, edge_index, edge_attr = datos.x, datos.edge_index, datos.edge_attr
        h = self.proyeccion_inicial(x)

        for gat, norm in zip(self.capas_gat, self.norm_capas):
            h_residual = h
            h = gat(h, edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = norm(h + h_residual)

        nodos_origen = h[edge_index[0]]
        nodos_destino = h[edge_index[1]]
        pareja_nodos = torch.cat([nodos_origen, nodos_destino], dim=1)
        
        return self.clasificador_arista(pareja_nodos)