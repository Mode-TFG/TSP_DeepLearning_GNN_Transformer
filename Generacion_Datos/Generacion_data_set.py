import torch
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

def calcular_matriz_distancias(coordenadas):
    """Calcula la matriz de distancias euclidianas usando broadcasting de NumPy."""
    diff = coordenadas[:, np.newaxis, :] - coordenadas[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=-1)

def resolver_tsp_ortools(coordenadas, limite_segundos=1):
    """Resuelve el TSP usando OR-Tools y devuelve la ruta y la matriz de adyacencia."""
    num_nodos = len(coordenadas)
    
    # Calcular distancias escaladas a enteros para la API de OR-Tools
    coords_escaladas = (coordenadas * 10000)
    distancias = calcular_matriz_distancias(coords_escaladas).astype(int)

    manager = pywrapcp.RoutingIndexManager(num_nodos, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distancias[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(limite_segundos)

    solucion = routing.SolveWithParameters(search_parameters)

    ruta = []
    if solucion:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            ruta.append(manager.IndexToNode(index))
            index = solucion.Value(routing.NextVar(index))
    
    # Crear matriz de adyacencia binaria
    adj_matrix = np.zeros((num_nodos, num_nodos), dtype=np.float32)
    for i in range(len(ruta)):
        nodo_actual = ruta[i]
        nodo_siguiente = ruta[(i + 1) % len(ruta)]
        adj_matrix[nodo_actual, nodo_siguiente] = 1.0
        adj_matrix[nodo_siguiente, nodo_actual] = 1.0 

    return ruta, adj_matrix

def generar_datasets_escalables(num_mapas=500, dimensiones=[20, 50, 100]):
    for num_ciudades in dimensiones:
        print(f"\n--- Iniciando generación para N={num_ciudades} ciudades ---")
        archivo_salida = f"dataset_test_universal_{num_ciudades}.pt"
        
        # Ajuste de tolerancia temporal para el Oráculo
        tiempo_oraculo = 1 if num_ciudades == 20 else (2 if num_ciudades == 50 else 5)
        
        inicio = time.time()
        dataset = []

        for i in range(num_mapas):
            coordenadas = np.random.rand(num_ciudades, 2).astype(np.float32)
            
            # Matriz en coma flotante puro para las redes neuronales
            matriz_dist = calcular_matriz_distancias(coordenadas).astype(np.float32)
            
            ruta_optima, matriz_adj = resolver_tsp_ortools(coordenadas, limite_segundos=tiempo_oraculo)
            
            mapa_dict = {
                'coordenadas': torch.tensor(coordenadas),
                'matriz_distancias': torch.tensor(matriz_dist),
                'matriz_adyacencia': torch.tensor(matriz_adj),
                'ruta_optima': torch.tensor(ruta_optima, dtype=torch.long)
            }
            dataset.append(mapa_dict)

            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{num_mapas}] resueltos para N={num_ciudades}...")

        torch.save(dataset, archivo_salida)
        print(f"Dataset guardado: {archivo_salida}. Tiempo invertido: {time.time() - inicio:.2f} s")

if __name__ == "__main__":
    generar_datasets_escalables(num_mapas=500, dimensiones=[20, 50, 100])