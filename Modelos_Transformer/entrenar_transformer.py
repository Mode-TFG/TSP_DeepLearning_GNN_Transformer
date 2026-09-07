"""
Entrenamiento del Transformer para el TSP mediante REINFORCE (Aprendizaje
por Refuerzo), con linea base de "rollout" voraz (greedy rollout baseline,
Kool et al. 2019) en vez de una media movil exponencial.

Idea: en vez de comparar cada ruta muestreada contra un escalar (la media
historica reciente), se compara contra la ruta que produce, PARA ESE MISMO
MAPA, una copia congelada de la red en modo voraz (determinista). Da una
ventaja por instancia, con mucha menos varianza que una linea base escalar.

La copia congelada ("baseline") solo se actualiza cuando la politica actual
la supera de forma estadisticamente significativa (test t pareado de una
cola, alpha=0.05) sobre un conjunto de validacion FIJO. Evita perseguir
ruido: el baseline solo "asciende" cuando hay evidencia real de mejora.

Requiere: pip install scipy pandas (si no estan ya instalados)
"""

import copy
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from scipy import stats

from TSP_TRANSFORMERS import TSPEncoder, TSPDecoder, calcular_longitud_ruta

# --- Semilla global -----------------------------------------------
torch.manual_seed(0)

# --- Configuracion del experimento ---------------------------------
BATCH_SIZE = 128
N_CIUDADES = 20
ITERACIONES = 500              # ver PASO 3 del plan: calibrar antes de fijarlo
LR = 1e-4
TAMANO_VALIDACION = 1000       # conjunto FIJO, generado una sola vez
INTERVALO_ROLLOUT = 50         # cada cuantas iteraciones se compara
ALPHA_TTEST = 0.05             # umbral de significancia (test de una cola)
CARPETA_FIGURAS = "figuras_tfg"
os.makedirs(CARPETA_FIGURAS, exist_ok=True)


def evaluar_voraz(encoder, decoder, mapas):
    """Forward determinista (greedy) y sin gradiente. Se usa tanto para la
    baseline por instancia dentro del bucle como para la comparacion
    periodica sobre el conjunto de validacion.
    """
    with torch.no_grad():
        embeddings = encoder(mapas)
        rutas, _ = decoder(embeddings, greedy=True)
        return calcular_longitud_ruta(mapas, rutas)


if __name__ == "__main__":
    # --- Red de politica (se entrena) y red baseline (copia congelada) --
    encoder = TSPEncoder()
    decoder = TSPDecoder()

    encoder_baseline = copy.deepcopy(encoder)
    decoder_baseline = copy.deepcopy(decoder)
    encoder_baseline.eval()
    decoder_baseline.eval()
    for p in list(encoder_baseline.parameters()) + list(decoder_baseline.parameters()):
        p.requires_grad_(False)

    optimizador = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

    # Conjunto de validacion FIJO: se genera una unica vez y se reutiliza
    # en todas las comparaciones -- permite un test PAREADO (mismos mapas
    # para politica y baseline en cada chequeo).
    mapas_validacion = torch.rand((TAMANO_VALIDACION, N_CIUDADES, 2))

    historial_costes = []       # coste medio de CADA lote de entrenamiento (ruidoso)
    historial_validacion = []   # coste medio politica/baseline en cada chequeo (limpio)

    print(f"Iniciando entrenamiento por REINFORCE + rollout baseline "
          f"({ITERACIONES} iteraciones, lote {BATCH_SIZE}, "
          f"validacion cada {INTERVALO_ROLLOUT} iters sobre {TAMANO_VALIDACION} mapas fijos)...")
    tiempo_inicio = time.time()

    for iteracion in range(ITERACIONES):
        grafos_batch = torch.rand((BATCH_SIZE, N_CIUDADES, 2))

        # Politica: forward ESTOCASTICO, con gradiente
        embeddings = encoder(grafos_batch)
        rutas, log_probs = decoder(embeddings, greedy=False)
        costes_politica = calcular_longitud_ruta(grafos_batch, rutas)

        # Baseline: forward VORAZ sobre el MISMO lote, sin gradiente.
        costes_baseline = evaluar_voraz(encoder_baseline, decoder_baseline, grafos_batch)

        # Ventaja = coste_politica - coste_baseline. Mismo signo que la
        # version EMA anterior (positiva = peor que la referencia = el
        # gradiente reduce su probabilidad); solo cambia que "baseline"
        # ahora es por instancia, no un escalar historico.
        ventaja = costes_politica - costes_baseline
        loss = (ventaja.detach() * log_probs.sum(dim=1)).mean()

        optimizador.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizador.step()

        historial_costes.append(costes_politica.mean().item())

        # --- Chequeo periodico: ¿la politica ya supera al baseline? -----
        if (iteracion + 1) % INTERVALO_ROLLOUT == 0:
            encoder.eval()
            decoder.eval()
            costes_politica_val = evaluar_voraz(encoder, decoder, mapas_validacion)
            costes_baseline_val = evaluar_voraz(encoder_baseline, decoder_baseline, mapas_validacion)
            encoder.train()
            decoder.train()

            # diferencias > 0 significa que la politica es mas barata (mejor)
            diferencias = (costes_baseline_val - costes_politica_val).detach().numpy()
            media_diferencia = diferencias.mean()

            if media_diferencia > 0:
                _, p_valor_dos_colas = stats.ttest_1samp(diferencias, popmean=0.0)
                p_valor_una_cola = p_valor_dos_colas / 2
                asciende = p_valor_una_cola < ALPHA_TTEST
                estado = ('BASELINE ACTUALIZADO' if asciende
                          else f'mejora no significativa (p={p_valor_una_cola:.4f})')
            else:
                asciende = False
                p_valor_una_cola = float('nan')
                estado = 'la politica no supera al baseline en este chequeo'

            print(f"Iteracion {iteracion + 1:03d}/{ITERACIONES} | "
                  f"politica={costes_politica_val.mean():.4f} | "
                  f"baseline={costes_baseline_val.mean():.4f} | {estado}")

            historial_validacion.append({
                'iteracion': iteracion + 1,
                'coste_politica': costes_politica_val.mean().item(),
                'coste_baseline': costes_baseline_val.mean().item(),
                'p_valor_una_cola': p_valor_una_cola,
                'actualizado': asciende,
            })

            if asciende:
                encoder_baseline.load_state_dict(encoder.state_dict())
                decoder_baseline.load_state_dict(decoder.state_dict())

    tiempo_total = time.time() - tiempo_inicio
    minutos, segundos = divmod(tiempo_total, 60)
    print("-" * 60)
    print(f"Entrenamiento completado en {int(minutos)} min {segundos:.2f} s.")
    n_actualizaciones = sum(1 for h in historial_validacion if h['actualizado'])
    print(f"El baseline se actualizo {n_actualizaciones}/{len(historial_validacion)} veces.")
    print("-" * 60)

    # --- Figura para la memoria: politica vs. baseline -------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, ITERACIONES + 1), historial_costes,
            color="#a6c8ff", linewidth=0.8, label="Coste por lote (bruto)")

    iters_val = [h['iteracion'] for h in historial_validacion]
    costes_pol_val = [h['coste_politica'] for h in historial_validacion]
    costes_base_val = [h['coste_baseline'] for h in historial_validacion]
    ax.plot(iters_val, costes_pol_val, color="#1f77b4", marker='o', markersize=3,
            label="Politica (validacion)")
    ax.plot(iters_val, costes_base_val, color="#d62728", marker='s', markersize=3,
            linestyle='--', label="Baseline (validacion)")

    ax.set_xlabel("Iteracion")
    ax.set_ylabel("Coste medio (longitud de ruta)")
    ax.set_title("Entrenamiento del Transformer: politica vs. rollout baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CARPETA_FIGURAS}/curva_entrenamiento_transformer.pdf", dpi=150)
    plt.show()

    pd.DataFrame(historial_validacion).to_csv(
        f"{CARPETA_FIGURAS}/tabla_rollout_baseline.csv", index=False)

    # --- Persistencia del modelo -----------------------------------------
    ruta_guardado = "transformer_tsp20_pesos.pth"
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'coste_final': historial_costes[-1],
        'historial_costes': historial_costes,
        'historial_validacion': historial_validacion,
    }, ruta_guardado)
    print(f"Arquitectura y pesos sinapticos guardados en: {ruta_guardado}")
