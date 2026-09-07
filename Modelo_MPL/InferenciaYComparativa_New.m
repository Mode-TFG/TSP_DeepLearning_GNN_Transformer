% =========================================================================
% MÓDULO 3.3: INFERENCIA, DECODIFICACIÓN Y EVIDENCIA DE FRACASO (MLP)
% =========================================================================
clc; clear; close all;

disp('Cargando modelo neuronal y generando mapa de examen...');
% 1. Cargar la red entrenada (Renombrada correctamente a MLP)
if ~isfile('RedMLP_TSP_Entrenada.mat')
    error('Falta la red entrenada. Ejecuta el entrenamiento del MLP primero.');
end
load('RedMLP_TSP_Entrenada.mat', 'redMLP_TSP', 'numCiudades');

% 2. Generar un mapa de prueba NUEVO (Datos fuera de la muestra)
rng('shuffle'); % Semilla aleatoria nueva
ciudadesTest = rand(numCiudades, 2);
distanciasTest = pdist2(ciudadesTest, ciudadesTest);

% Normalizar entrada tal y como aprendió la red
X_test = reshape(distanciasTest / max(distanciasTest(:)), [1, 1, numCiudades^2, 1]);

% =========================================================================
% FASE 1: ÓPTIMO MATEMÁTICO (MILP - LA VERDAD FUNDAMENTAL)
% =========================================================================
disp('1. Resolviendo mapa de forma exacta (MILP)...');
tic;
rutaMILP = resolverTSPExacto(ciudadesTest, distanciasTest);
tiempoMILP = toc;
costeMILP = calcularCoste(rutaMILP, distanciasTest);

% =========================================================================
% FASE 2: LÍNEA BASE (HEURÍSTICA DEL VECINO MÁS CERCANO)
% =========================================================================
disp('2. Resolviendo mapa con heurística voraz...');
tic;
rutaNN = zeros(1, numCiudades);
nodoActual = 1; 
visitadosNN = false(1, numCiudades);
rutaNN(1) = nodoActual; 
visitadosNN(nodoActual) = true;
for i = 2:numCiudades
    dist = distanciasTest(nodoActual, :);
    dist(visitadosNN) = inf; 
    [~, nodoSiguiente] = min(dist); 
    rutaNN(i) = nodoSiguiente;
    nodoActual = nodoSiguiente;
    visitadosNN(nodoActual) = true;
end
tiempoNN = toc;
costeNN = calcularCoste(rutaNN, distanciasTest);

% =========================================================================
% FASE 3: LA INTELIGENCIA ARTIFICIAL (PERCEPTRÓN MULTICAPA)
% =========================================================================
disp('3. Realizando inferencia y decodificación difusa (MLP)...');
% A. Inferencia: Que la red hable
tic;
Y_pred_1D = predict(redMLP_TSP, X_test);
tiempoInferencia = toc;

% Reconstruir la matriz de probabilidades NxN
matrizProbabilidades = reshape(Y_pred_1D, [numCiudades, numCiudades]);
matrizProbabilidades = (matrizProbabilidades + matrizProbabilidades') / 2;
matrizProbabilidades(1:numCiudades+1:end) = 0; % Diagonal a 0

% B. Decodificación Voraz: Convertir probabilidad en ruta válida
tic;
rutaMLP = zeros(1, numCiudades);
nodoActual = 1;
visitadosMLP = false(1, numCiudades);
rutaMLP(1) = nodoActual;
visitadosMLP(nodoActual) = true;
for i = 2:numCiudades
    probabilidadesActual = matrizProbabilidades(nodoActual, :);
    probabilidadesActual(visitadosMLP) = -inf; % Restricción anti-subtours
    
    [~, nodoSiguiente] = max(probabilidadesActual); % Nodo con mayor prob
    
    rutaMLP(i) = nodoSiguiente;
    nodoActual = nodoSiguiente;
    visitadosMLP(nodoActual) = true;
end
tiempoDecodificacion = toc;
tiempoTotalMLP = tiempoInferencia + tiempoDecodificacion;
costeMLP = calcularCoste(rutaMLP, distanciasTest);

% =========================================================================
% MÓDULO VISUAL: COMPARATIVA DE RUTAS (LaTeX Export)
% =========================================================================
disp('Generando figura de comparativa geométrica (Contraste corregido)...');
fig = figure('Position', [100, 100, 1200, 400], 'Color', 'w');

rutas = {rutaMILP, rutaNN, rutaMLP};
titulos = {'A. Óptimo Matemático (MILP)', 'B. Heurística (NN)', 'C. Perceptrón (MLP)'};
colores = {[0.4660 0.6740 0.1880], [0 0.4470 0.7410], [0.8500 0.3250 0.0980]};

for p = 1:3
    subplot(1, 3, p); hold on; grid on;
    rt = rutas{p}; rt = [rt, rt(1)]; % Cerrar polígono
    
    % Trazado de las rutas
    plot(ciudadesTest(rt, 1), ciudadesTest(rt, 2), '-', 'Color', colores{p}, 'LineWidth', 1.5);
    
    % Nodos con corrección de oclusión y bordes negros puros
    scatter(ciudadesTest(:,1), ciudadesTest(:,2), 40, 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
    
    % Textos y ejes forzados a negro
    title(['\bf ', titulos{p}], 'Interpreter', 'tex', 'FontSize', 12, 'Color', 'k');
    axis equal; axis([0 1 0 1]); box on;
    set(gca, 'TickLabelInterpreter', 'tex', 'FontSize', 10, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 0.8);
    xlabel('coordenada X', 'Interpreter', 'tex', 'Color', 'k'); 
    ylabel('coordenada Y', 'Interpreter', 'tex', 'Color', 'k');
end

exportgraphics(fig, 'comparativa_rutas_mlp.pdf', 'ContentType', 'vector');
disp('Gráfico guardado como "comparativa_rutas_mlp.pdf".');

% =========================================================================
% REPORTE DE RESULTADOS Y GENERACIÓN DE CÓDIGO LATEX
% =========================================================================
gapNN = ((costeNN - costeMILP) / costeMILP) * 100;
gapMLP = ((costeMLP - costeMILP) / costeMILP) * 100;

disp(' ');
disp('======================================================');
disp('   CÓDIGO LATEX GENERADO: TABLA 4.2 (RENDIMIENTO)     ');
disp('======================================================');
fprintf('\\begin{table}[htbp]\n');
fprintf('    \\centering\n');
fprintf('    \\caption{Métricas de rendimiento computacional y brecha de optimalidad entre arquitecturas.}\n');
fprintf('    \\label{tab:metricas_rendimiento}\n');
fprintf('    \\begin{tabular}{lccc}\n');
fprintf('        \\toprule\n');
fprintf('        \\textbf{Modelo / Algoritmo} & \\textbf{Tiempo (s)} & \\textbf{Distancia Total} & \\textbf{Brecha Optimalidad (\\%%)} \\\\\n');
fprintf('        \\midrule\n');
fprintf('        Óptimo (MILP) & %.4f & %.4f & 0.00 \\\\\n', tiempoMILP, costeMILP);
fprintf('        Oráculo Heurístico (NN) & %.4f & %.4f & +%.2f \\\\\n', tiempoNN, costeNN, gapNN);
fprintf('        Perceptrón Multicapa (MLP) & %.4f & %.4f & +%.2f \\\\\n', tiempoTotalMLP, costeMLP, gapMLP);
fprintf('        \\bottomrule\n');
fprintf('    \\end{tabular}\n');
fprintf('\\end{table}\n');
disp('======================================================');

% =========================================================================
% FUNCIONES AUXILIARES
% =========================================================================
function coste = calcularCoste(ruta, distancias)
    n = length(ruta); coste = 0;
    for k = 1:(n-1)
        coste = coste + distancias(ruta(k), ruta(k+1));
    end
    coste = coste + distancias(ruta(end), ruta(1));
end

function ruta = resolverTSPExacto(ciudades, distancias)
    n = size(ciudades, 1);
    viajes = nchoosek(1:n, 2);
    numViajes = size(viajes, 1);
    distanciasViajes = zeros(numViajes, 1);
    for k = 1:numViajes
        distanciasViajes(k) = distancias(viajes(k,1), viajes(k,2));
    end
    Aeq = spalloc(n, numViajes, n*(n-1));
    for k = 1:numViajes
        Aeq(viajes(k,1), k) = 1; Aeq(viajes(k,2), k) = 1;
    end
    beq = 2 * ones(n, 1);
    intcon = 1:numViajes; lb = zeros(numViajes, 1); ub = ones(numViajes, 1);
    opciones = optimoptions('intlinprog', 'Display', 'off');
    Aineq = []; bineq = []; numSubtours = 2;
    while numSubtours > 1
        [x_opt, ~] = intlinprog(distanciasViajes, intcon, Aineq, bineq, Aeq, beq, lb, ub, opciones);
        aristasSeleccionadas = find(round(x_opt) == 1);
        G = graph(viajes(aristasSeleccionadas, 1), viajes(aristasSeleccionadas, 2));
        [bins, ~] = conncomp(G);
        numSubtours = max(bins);
        if numSubtours > 1
            for k = 1:numSubtours
                nodosSubtour = find(bins == k);
                idxSubtour = ismember(viajes(:,1), nodosSubtour) & ismember(viajes(:,2), nodosSubtour);
                nuevaA = zeros(1, numViajes); nuevaA(idxSubtour) = 1;
                Aineq = [Aineq; nuevaA]; bineq = [bineq; length(nodosSubtour) - 1];
            end
        end
    end
    ruta = zeros(1, n); nodoActual = 1; ruta(1) = nodoActual;
    for i = 2:n
        vecinos = neighbors(G, nodoActual);
        if i == 2; nodoSiguiente = vecinos(1);
        else; nodoSiguiente = vecinos(vecinos ~= ruta(i-2)); end
        ruta(i) = nodoSiguiente; nodoActual = nodoSiguiente;
    end
end