import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from TSP_GNN import RedNeuronalTSP

torch.manual_seed(0)
np.random.seed(0)

ciudades_entrenamiento = 20
num_mapas_entrenamiento = 5000
tamaño_lote = 64
lr_final = 0.005
epocas_warmup = 10
epocas = 90

carpeta_figuras = "figuras_tfg"
os.makedirs(carpeta_figuras, exist_ok=True)

def generar_mapa_tsp(num_ciudades):
    coordenadas = torch.rand((num_ciudades, 2), dtype=torch.float)
    nodos = torch.arange(num_ciudades)
    origen, destino = torch.meshgrid(nodos, nodos, indexing='ij')
    origen = origen.flatten()
    destino = destino.flatten()

    mascara = origen != destino
    edge_index = torch.stack([origen[mascara], destino[mascara]], dim=0)

    puntos_origen = coordenadas[edge_index[0]]
    puntos_destino = coordenadas[edge_index[1]]
    distancias = torch.norm(puntos_origen - puntos_destino, p=2, dim=1)
    edge_attr = distancias.view(-1, 1)

    return Data(x=coordenadas, edge_index=edge_index, edge_attr=edge_attr)

def solucionador_ortools(mapa_grafo):
    coords_np = mapa_grafo.x.numpy() * 10000
    num_nodos = len(coords_np)
    matriz_dist = np.zeros((num_nodos, num_nodos), dtype=int)
    for i in range(num_nodos):
        for j in range(num_nodos):
            matriz_dist[i][j] = int(np.linalg.norm(coords_np[i] - coords_np[j]))

    manager = pywrapcp.RoutingIndexManager(num_nodos, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def funcion_distancia(desde_indice, hacia_indice):
        return matriz_dist[manager.IndexToNode(desde_indice)][manager.IndexToNode(hacia_indice)]

    transit_callback_index = routing.RegisterTransitCallback(funcion_distancia)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    parametros = pywrapcp.DefaultRoutingSearchParameters()
    parametros.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parametros.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parametros.time_limit.seconds = 1

    solucion = routing.SolveWithParameters(parametros)
    ruta_optima = []
    indice = routing.Start(0)
    while not routing.IsEnd(indice):
        ruta_optima.append(manager.IndexToNode(indice))
        indice = solucion.Value(routing.NextVar(indice))
    ruta_optima.append(manager.IndexToNode(indice))

    etiquetas_aristas = torch.zeros(mapa_grafo.num_edges, 1)
    pares_optimos = set()
    for i in range(len(ruta_optima) - 1):
        pares_optimos.add((ruta_optima[i], ruta_optima[i + 1]))
        pares_optimos.add((ruta_optima[i + 1], ruta_optima[i]))

    for idx in range(mapa_grafo.num_edges):
        if (mapa_grafo.edge_index[0, idx].item(), mapa_grafo.edge_index[1, idx].item()) in pares_optimos:
            etiquetas_aristas[idx] = 1.0
    return etiquetas_aristas

if __name__ == "__main__":
    print(f"Generando {num_mapas_entrenamiento} mapas de entrenamiento...")
    dataset_entrenamiento = []
    for i in range(num_mapas_entrenamiento):
        mapa = generar_mapa_tsp(ciudades_entrenamiento)
        mapa.y = solucionador_ortools(mapa)
        dataset_entrenamiento.append(mapa)

    cargador_entrenamiento = DataLoader(dataset_entrenamiento, batch_size=tamaño_lote, shuffle=True)
    total_positivos = sum(g.y.sum().item() for g in dataset_entrenamiento)
    peso_castigo = torch.tensor([(sum(g.num_edges for g in dataset_entrenamiento) - total_positivos) / total_positivos])

    modelo_ia = RedNeuronalTSP()
    optimizador = optim.Adam(modelo_ia.parameters(), lr=lr_final)
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso_castigo)

    def lr_en_epoca(epoca):
        lr_inicial = lr_final / 10
        return lr_inicial + (lr_final - lr_inicial) * (epoca / epocas_warmup) if epoca <= epocas_warmup else lr_final

    historial_perdida = []
    print("Iniciando entrenamiento...")
    for epoca in range(1, epocas + 1):
        lr_actual = lr_en_epoca(epoca)
        for grupo in optimizador.param_groups:
            grupo['lr'] = lr_actual

        modelo_ia.train()
        perdida_acumulada = 0
        for lote in cargador_entrenamiento:
            optimizador.zero_grad()
            perdida = criterio(modelo_ia(lote), lote.y)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += perdida.item()
        
        historial_perdida.append(perdida_acumulada / len(cargador_entrenamiento))
        if epoca % 10 == 0:
            print(f"Época {epoca}/{epocas} - Pérdida: {historial_perdida[-1]:.4f}")

    torch.save(modelo_ia.state_dict(), "modelo_gnn.pt")
    print("Modelo guardado en 'modelo_gnn.pt'")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, epocas + 1), historial_perdida, color="#1f77b4")
    ax.axvspan(1, epocas_warmup, color="orange", alpha=0.15, label="Warm-up del lr")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida media (BCE ponderada)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{carpeta_figuras}/curva_perdida.png", dpi=150)