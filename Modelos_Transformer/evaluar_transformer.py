import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from numba import njit
from TSP_TRANSFORMERS import TSPEncoder, TSPDecoder, calcular_longitud_ruta

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

if __name__ == "__main__":
    print("=== AUDITORÍA DE RENDIMIENTO: TRANSFORMER (GREEDY) + 2-OPT ===")
    carpeta_figuras = "figuras_tfg"
    os.makedirs(carpeta_figuras, exist_ok=True)

    encoder = TSPEncoder()
    decoder = TSPDecoder()
    checkpoint = torch.load("Transformer/transformer_tsp20_pesos.pth", weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()

    datasets = [
        ("dataset_test_universal_20.pt", "20 nodos", 20),
        ("dataset_test_universal_50.pt", "50 nodos", 50),
        ("dataset_test_universal_100.pt", "100 nodos", 100)
    ]

    filas_resultados = []

    with torch.no_grad():
        for ruta_ds, etiqueta, ciudades in datasets:
            if not os.path.exists(ruta_ds): continue
            dataset_test = torch.load(ruta_ds, weights_only=False)
            
            gaps_crudo, gaps_2opt = [], []
            tiempos_inferencia, tiempos_2opt = [], []

            for i, mapa in enumerate(dataset_test):
                coordenadas = mapa['coordenadas']
                matriz_dist = mapa['matriz_distancias']
                
                # Expandimos a 1 muestra al ser decodificación voraz (Greedy)
                coordenadas_batch = coordenadas.unsqueeze(0)

                # --- 1. Inferencia Pura del Transformer ---
                t0_inf = time.time()
                embeddings = encoder(coordenadas_batch)
                rutas_predichas, _ = decoder(embeddings, greedy=True)
                costes_batch = calcular_longitud_ruta(coordenadas_batch, rutas_predichas)
                mejor_coste, indice_mejor = torch.min(costes_batch, dim=0)
                ruta_predicha_optima = rutas_predichas[indice_mejor].tolist()
                t_inf_total = (time.time() - t0_inf) * 1000

                # Extracción del gap crudo
                ruta_referencia = mapa['ruta_optima'].tolist()
                coste_optimo = sum(matriz_dist[ruta_referencia[j], ruta_referencia[(j + 1) % len(ruta_referencia)]].item() for j in range(len(ruta_referencia)))
                gap_c = (mejor_coste.item() - coste_optimo) / coste_optimo * 100

                # --- 2. Post-procesamiento 2-Opt ---
                t0_2opt = time.time()
                ruta_refinada = dos_opt(ruta_predicha_optima, coordenadas)
                t_2opt_total = (time.time() - t0_2opt) * 1000

                ruta_refinada_batch = torch.tensor(ruta_refinada).unsqueeze(0)
                coste_2opt_val = calcular_longitud_ruta(coordenadas.unsqueeze(0), ruta_refinada_batch).item()
                gap_2 = (coste_2opt_val - coste_optimo) / coste_optimo * 100

                # --- 3. Filtro de Registro ---
                # Excluimos la primera iteración (i == 0) para evitar que la compilación JIT
                # de Numba y la asignación de memoria de PyTorch contaminen la varianza.
                if i > 0:
                    tiempos_inferencia.append(t_inf_total)
                    tiempos_2opt.append(t_2opt_total)
                    gaps_crudo.append(gap_c)
                    gaps_2opt.append(gap_2)

            print(f"\n--- {etiqueta} ---")
            print(f"Gap Transformer: {np.mean(gaps_crudo):.2f} ± {np.std(gaps_crudo):.2f} %")
            print(f"Gap Híbrido:     {np.mean(gaps_2opt):.2f} ± {np.std(gaps_2opt):.2f} %")
            print(f"T. Inferencia:   {np.mean(tiempos_inferencia):.2f} ± {np.std(tiempos_inferencia):.2f} ms")
            print(f"T. 2-Opt:        {np.mean(tiempos_2opt):.2f} ± {np.std(tiempos_2opt):.2f} ms")

            filas_resultados.append({
                "Escenario": etiqueta,
                "Gap Transformer (%)": np.mean(gaps_crudo),
                "Gap Trans. Std": np.std(gaps_crudo),
                "Gap +2Opt (%)": np.mean(gaps_2opt),
                "Gap +2Opt Std": np.std(gaps_2opt),
                "T. Inferencia (ms)": np.mean(tiempos_inferencia),
                "T. Inf. Std (ms)": np.std(tiempos_inferencia),
                "T. +2Opt (ms)": np.mean(tiempos_2opt),
                "T. 2Opt Std (ms)": np.std(tiempos_2opt)
            })

    if filas_resultados:
        df = pd.DataFrame(filas_resultados).round(2)
        df.to_csv(f"{carpeta_figuras}/tabla_evaluacion_transformer_greedy.csv", index=False)
        
        escenarios = df["Escenario"]
        x = np.arange(len(escenarios))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bar1 = ax.bar(x - width/2, df["Gap Transformer (%)"], width, label='Greedy Puro', color='#d62728')
        bar2 = ax.bar(x + width/2, df["Gap +2Opt (%)"], width, label='Con 2-Opt', color='#2ca02c')
        ax.set_ylabel('Gap Medio frente a Referencia (%)')
        ax.set_title('Degradación Topológica: Transformer (Greedy) vs Híbrido')
        ax.set_xticks(x)
        ax.set_xticklabels(escenarios)
        ax.legend()
        ax.bar_label(bar1, fmt="%.2f%%", padding=3)
        ax.bar_label(bar2, fmt="%.2f%%", padding=3)
        plt.tight_layout()
        plt.savefig(f"{carpeta_figuras}/grafica_gaps_transformer_greedy.png", dpi=150)
        plt.close()