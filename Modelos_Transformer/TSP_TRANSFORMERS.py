"""
Arquitectura Transformer para el TSP: codificador de auto-atencion +
decodificador autorregresivo tipo puntero (pointer network).

Ambas piezas se combinan para producir, dada una nube de N ciudades en 2D,
una permutacion completa (ciclo Hamiltoniano) de las N ciudades. El
decodificador garantiza rutas validas por construccion: enmascara las
ciudades ya visitadas, asi que nunca puede repetir ni omitir un nodo
(a diferencia del decodificador de Kruskal del capitulo de la GNN, que
si podia fallar en cerrar el ciclo).
"""

import torch
import torch.nn as nn


class TSPEncoder(nn.Module):
    """Codificador: proyecta coordenadas 2D a un espacio latente y las
    contextualiza mediante auto-atencion estandar.

    Deliberadamente SIN codificacion posicional: las ciudades de una
    instancia del TSP no tienen un orden intrinseco -- el problema es
    invariante a permutaciones de la entrada. Anadir un embedding
    posicional (como en NLP) rompería esa invarianza y sesgaria al
    modelo hacia el orden arbitrario en que se listan las coordenadas.
    """

    def __init__(self, embedding_dim=128, num_heads=8, num_layers=3):
        super(TSPEncoder, self).__init__()

        self.embedding = nn.Linear(2, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, coordenadas):
        """coordenadas: (Batch, N, 2) -> embeddings: (Batch, N, embedding_dim)"""
        h = self.embedding(coordenadas)
        return self.transformer_encoder(h)


class TSPDecoder(nn.Module):
    """Decodificador autorregresivo tipo puntero.

    En cada paso, calcula una consulta de atencion a partir de TRES
    senales de contexto -- el embedding medio del grafo, el embedding
    de la ciudad de PARTIDA y el de la ciudad ACTUAL -- y la compara
    (producto escalar) contra el embedding de cada ciudad para obtener
    una distribucion de probabilidad sobre "la siguiente ciudad a
    visitar". Las ciudades ya visitadas se enmascaran a -inf antes del
    softmax, lo que garantiza por construccion que la ruta resultante
    sea una permutacion valida.

    Incluir el embedding de la ciudad de partida en el contexto (ademas
    del de la ciudad actual) sigue la formulacion del Attention Model de
    Kool et al. (2019): sin esa senal, el modelo no tiene forma de saber
    a que nodo debe volver para cerrar el ciclo de forma barata, y tiende
    a cometer errores caros en el ultimo tramo de la ruta.
    """

    def __init__(self, embedding_dim=128):
        super(TSPDecoder, self).__init__()
        self.embedding_dim = embedding_dim

        # 3 senales de contexto concatenadas: grafo + nodo inicial + nodo actual
        self.W_q = nn.Linear(embedding_dim * 3, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.nodo_inicial_dummy = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, embeddings_globales, greedy=False):
        batch_size, n_ciudades, d = embeddings_globales.shape

        contexto_global = embeddings_globales.mean(dim=1, keepdim=True)
        keys = self.W_k(embeddings_globales)
        mascara = torch.ones((batch_size, n_ciudades), dtype=torch.bool, device=embeddings_globales.device)

        rutas = []
        log_probs = []
        nodo_actual = self.nodo_inicial_dummy.expand(batch_size, -1, -1)
        nodo_primero = None  # se fija tras elegir la primera ciudad real

        for step in range(n_ciudades):
            # Antes de elegir la primera ciudad no hay "nodo inicial" todavia:
            # se usa el mismo marcador dummy que nodo_actual en ese paso.
            referencia_inicio = nodo_primero if nodo_primero is not None else nodo_actual
            input_query = torch.cat([contexto_global, referencia_inicio, nodo_actual], dim=-1)
            query = self.W_q(input_query)

            scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / (d ** 0.5)
            scores_enmascarados = scores.clone()
            scores_enmascarados[~mascara] = float('-inf')

            probs = torch.softmax(scores_enmascarados, dim=1)

            # EL CONTROL DE MODO: Exploracion (Train) vs Explotacion (Test)
            if greedy:
                ciudad_elegida = torch.argmax(probs, dim=1)
            else:
                distribucion = torch.distributions.Categorical(probs)
                ciudad_elegida = distribucion.sample()
                log_probs.append(distribucion.log_prob(ciudad_elegida))

            rutas.append(ciudad_elegida)
            mascara.scatter_(1, ciudad_elegida.unsqueeze(1), False)
            nodo_actual = embeddings_globales[torch.arange(batch_size), ciudad_elegida].unsqueeze(1)

            if nodo_primero is None:
                nodo_primero = nodo_actual

        rutas = torch.stack(rutas, dim=1)

        if greedy:
            return rutas, None  # no hacen falta log_probs en evaluacion
        else:
            log_probs = torch.stack(log_probs, dim=1)
            return rutas, log_probs


def calcular_longitud_ruta(coordenadas, rutas):
    """Longitud total (Ciclo Hamiltoniano) de un lote de rutas, vectorizada.

    Unica implementacion de esta cuenta en todo el proyecto Transformer:
    la usan tanto el script de entrenamiento (como recompensa por lote)
    como el de evaluacion (coste de una ruta individual, lote de tamano 1).

    coordenadas: (Batch, N, 2)
    rutas:       (Batch, N) -- indices de ciudad en orden de visita
    devuelve:    (Batch,)   -- longitud total de cada ruta, cierre incluido
    """
    batch_size, n_ciudades, _ = coordenadas.shape

    rutas_expandidas = rutas.unsqueeze(-1).expand(batch_size, n_ciudades, 2)
    coordenadas_ordenadas = torch.gather(coordenadas, 1, rutas_expandidas)

    # torch.roll cierra el ciclo: conecta la ultima ciudad visitada con la primera
    coordenadas_siguientes = torch.roll(coordenadas_ordenadas, shifts=-1, dims=1)

    distancias_aristas = torch.norm(coordenadas_siguientes - coordenadas_ordenadas, p=2, dim=2)
    return distancias_aristas.sum(dim=1)


if __name__ == "__main__":
    # Prueba de humo: verifica formas de tensor, no rendimiento del modelo.
    # No requiere pesos guardados ni datasets -- se puede ejecutar tal cual.
    BATCH_SIZE, N_CIUDADES, EMBEDDING_DIM = 16, 20, 128

    encoder = TSPEncoder(embedding_dim=EMBEDDING_DIM)
    decoder = TSPDecoder(embedding_dim=EMBEDDING_DIM)

    mapas_dummy = torch.rand((BATCH_SIZE, N_CIUDADES, 2))
    embeddings = encoder(mapas_dummy)
    rutas, log_probs = decoder(embeddings)
    longitudes = calcular_longitud_ruta(mapas_dummy, rutas)

    print(f"Entrada:            {tuple(mapas_dummy.shape)}")
    print(f"Embeddings:         {tuple(embeddings.shape)}")
    print(f"Rutas:              {tuple(rutas.shape)}  (debe ser [{BATCH_SIZE}, {N_CIUDADES}])")
    print(f"Log-probs:          {tuple(log_probs.shape)}")
    print(f"Longitudes de ruta: {tuple(longitudes.shape)}")
