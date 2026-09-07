import os
import time
import torch
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from numba import njit
from torch_geometric.data import Data
from TSP_GNN import RedNeuronalTSP

carpeta_figuras = "figuras_tfg"
os.makedirs(carpeta_figuras, exist_ok=True)

def decodificar_ruta(modelo, mapa, ciudades):
    modelo.eval()
    with torch.no_grad():
        probabilidades = torch.sigmoid(modelo(mapa)).numpy().flatten()
    indices_ordenados = np.argsort(probabilidades)[::-1]
    aristas_ia, grados, componentes = [], {i: 0 for i in range(ciudades)}, {i: i for i in range(ciudades)}

    def encontrar_raiz(nodo):
        while componentes[nodo] != nodo: nodo = componentes[nodo]
        return nodo

    for idx in indices_ordenados:
        origen, destino = mapa.edge_index[0, idx].item(), mapa.edge_index[1, idx].item()
        if grados[origen] >= 2 or grados[destino] >= 2: continue
        r_orig, r_dest = encontrar_raiz(origen), encontrar_raiz(destino)
        if r_orig != r_dest or len(aristas_ia) == ciudades - 1:
            aristas_ia.append((origen, destino))
            grados[origen] += 1
            grados[destino] += 1
            componentes[r_orig] = r_dest
        if len(aristas_ia) == ciudades: break
    return aristas_ia

def aristas_a_ruta(aristas, ciudades):
    vecinos = {i: [] for i in range(ciudades)}
    for o, d in aristas:
        vecinos[o].append(d)
        vecinos[d].append(o)
    ruta, anterior, actual = [0], -1, 0
    for _ in range(ciudades - 1):
        siguiente = [v for v in vecinos[actual] if v != anterior][0]
        ruta.append(siguiente)
        anterior, actual = actual, siguiente
    return ruta

@njit
def dos_opt_numba(ruta_array, matriz_distancias):
    ruta = ruta_array.copy()
    n = len(ruta)
    mejora = True
    while mejora:
        mejora = False
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1: continue
                a, b = ruta[i], ruta[i + 1]
                c, d = ruta[j], ruta[(j + 1) % n]
                coste_actual = matriz_distancias[a, b] + matriz_distancias[c, d]
                coste_nuevo = matriz_distancias[a, c] + matriz_distancias[b, d]
                if coste_nuevo < coste_actual - 1e-9:
                    ruta[i + 1:j + 1] = ruta[i + 1:j + 1][::-1]
                    mejora = True
    return ruta

def dos_opt(ruta, coords):
    if hasattr(coords, 'cpu'): coords = coords.detach().cpu().numpy()
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    matriz_distancias = np.sqrt(np.sum(diff**2, axis=-1))
    ruta_array = np.array(ruta, dtype=np.int32)
    return dos_opt_numba(ruta_array, matriz_distancias).tolist()

def longitud_ruta(ruta, coords):
    n = len(ruta)
    return sum(np.linalg.norm(coords[ruta[i]] - coords[ruta[(i + 1) % n]]) for i in range(n))

def solucionador_ortools(mapa_grafo):
    coords_np = mapa_grafo.x.numpy() * 10000
    num_nodos = len(coords_np)
    matriz_dist = np.zeros((num_nodos, num_nodos), dtype=int)
    for i in range(num_nodos):
        for j in range(num_nodos):
            matriz_dist[i][j] = int(np.linalg.norm(coords_np[i] - coords_np[j]))
    manager = pywrapcp.RoutingIndexManager(num_nodos, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(
        lambda d, h: matriz_dist[manager.IndexToNode(d)][manager.IndexToNode(h)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    parametros = pywrapcp.DefaultRoutingSearchParameters()
    parametros.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parametros.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    tiempo_ms = int((num_nodos / 20.0) * 1000)
    parametros.time_limit.seconds = tiempo_ms // 1000
    parametros.time_limit.nanos = (tiempo_ms % 1000) * int(1e6)
    solucion = routing.SolveWithParameters(parametros)
    ruta = []
    if solucion:
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            ruta.append(manager.IndexToNode(idx))
            idx = solucion.Value(routing.NextVar(idx))
        ruta.append(manager.IndexToNode(idx))
    return ruta

if __name__ == "__main__":
    print("=== ESTADÍSTICAS GNN ===")
    modelo_ia = RedNeuronalTSP()
    modelo_ia.load_state_dict(torch.load("modelo_gnn.pt", weights_only=True))
    modelo_ia.eval()

    datasets = [
        ("dataset_test_universal_20.pt", "20 nodos", 20),
        ("dataset_test_universal_50.pt", "50 nodos", 50),
        ("dataset_test_universal_100.pt", "100 nodos", 100)
    ]

    filas_csv = []
    
    with torch.no_grad():
        for ruta_ds, etiqueta, ciudades in datasets:
            if not os.path.exists(ruta_ds): continue
            dataset_test = torch.load(ruta_ds, weights_only=False)
            gaps_pura, gaps_2opt, t_or, t_gnn, t_gnn2opt = [], [], [], [], []
            
            for i, datos_dic in enumerate(dataset_test):
                coords_tensor = datos_dic['coordenadas']
                nodos = torch.arange(ciudades)
                orig, dest = torch.meshgrid(nodos, nodos, indexing='ij')
                mascara = orig.flatten() != dest.flatten()
                edge_index = torch.stack([orig.flatten()[mascara], dest.flatten()[mascara]], dim=0)
                edge_attr = torch.norm(coords_tensor[edge_index[0]] - coords_tensor[edge_index[1]], p=2, dim=1).view(-1, 1)
                mapa = Data(x=coords_tensor, edge_index=edge_index, edge_attr=edge_attr)
                coords = coords_tensor.numpy()
                
                matriz_dist_ref = datos_dic['matriz_distancias']
                ruta_opt = datos_dic['ruta_optima'].tolist()
                l_opt = sum(matriz_dist_ref[ruta_opt[j], ruta_opt[(j + 1) % len(ruta_opt)]].item() for j in range(len(ruta_opt)))
                
                t0 = time.time()
                _ = solucionador_ortools(mapa)
                t_or_val = (time.time() - t0) * 1000

                t0_gnn = time.time()
                aristas_gnn = decodificar_ruta(modelo_ia, mapa, ciudades)
                t1_gnn = time.time()
                
                if len(aristas_gnn) == ciudades:
                    ruta_ciudades = aristas_a_ruta(aristas_gnn, ciudades)
                    l_gnn_pura = longitud_ruta(ruta_ciudades, coords)
                    
                    t0_2opt = time.time()
                    ruta_refinada = dos_opt(ruta_ciudades, coords)
                    t1_2opt = time.time()
                    l_gnn_2opt = longitud_ruta(ruta_refinada, coords)
                    
                    # Exclusión estricta de la compilación JIT y asignación en frío de PyTorch
                    if i > 0:
                        t_or.append(t_or_val)
                        t_gnn.append((t1_gnn - t0_gnn) * 1000)
                        t_gnn2opt.append((t1_gnn - t0_gnn + t1_2opt - t0_2opt) * 1000)
                        gaps_pura.append((l_gnn_pura - l_opt) / l_opt * 100)
                        gaps_2opt.append((l_gnn_2opt - l_opt) / l_opt * 100)
            
            print(f"\n--- {etiqueta} ---")
            print(f"Gap GNN:      {np.mean(gaps_pura):.2f} ± {np.std(gaps_pura):.2f} %")
            print(f"Gap +2Opt:    {np.mean(gaps_2opt):.2f} ± {np.std(gaps_2opt):.2f} %")
            print(f"T. OR-Tools:  {np.mean(t_or):.2f} ± {np.std(t_or):.2f} ms")
            print(f"T. GNN Pura:  {np.mean(t_gnn):.2f} ± {np.std(t_gnn):.2f} ms")
            print(f"T. GNN+2Opt:  {np.mean(t_gnn2opt):.2f} ± {np.std(t_gnn2opt):.2f} ms")

            filas_csv.append({
                "Escenario": etiqueta,
                "Gap GNN (%)": np.mean(gaps_pura),
                "Gap GNN Std": np.std(gaps_pura),
                "Gap +2Opt (%)": np.mean(gaps_2opt),
                "Gap +2Opt Std": np.std(gaps_2opt),
                "T. OR-Tools (ms)": np.mean(t_or),
                "T. OR-Tools Std (ms)": np.std(t_or),
                "T. GNN Pura (ms)": np.mean(t_gnn),
                "T. GNN Pura Std (ms)": np.std(t_gnn),
                "T. Total +2Opt (ms)": np.mean(t_gnn2opt),
                "T. Total +2Opt Std (ms)": np.std(t_gnn2opt)
            })
    
    if filas_csv:
        pd.DataFrame(filas_csv).round(2).to_csv(f"{carpeta_figuras}/tabla_evaluacion_estatica_desglosada.csv", index=False)