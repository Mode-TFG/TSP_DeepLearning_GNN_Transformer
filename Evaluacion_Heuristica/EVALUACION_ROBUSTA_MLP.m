% =========================================================================
% MÓDULO 3.3b: EVALUACIÓN ROBUSTA DEL MLP SOBRE VARIOS MAPAS DE PRUEBA
% =========================================================================
clc; clear; close all;

disp('Cargando modelo neuronal entrenado...');
if ~isfile('RedMLP_TSP_Entrenada.mat')
    error('Falta la red entrenada. Ejecuta el entrenamiento del MLP primero.');
end
load('RedMLP_TSP_Entrenada.mat', 'redMLP_TSP', 'numCiudades');

% --- Configuración de la evaluación ---
numMapasTest = 1000;   % empieza aquí; el MILP puede tardar -- ver nota abajo
rng('shuffle');       % semilla distinta cada vez que arrancas este script

gapsNN  = zeros(numMapasTest, 1);
gapsMLP = zeros(numMapasTest, 1);
tiemposMILP = zeros(numMapasTest, 1);

fprintf('Evaluando el MLP sobre %d mapas nuevos (%d ciudades cada uno)...\n', numMapasTest, numCiudades);
inicioTotal = tic;

for m = 1:numMapasTest
    % --- Mapa de prueba nuevo ---
    ciudadesTest = rand(numCiudades, 2);
    distanciasTest = pdist2(ciudadesTest, ciudadesTest);

    % --- Óptimo exacto (MILP) ---
    tic;
    rutaMILP = resolverTSPExacto(ciudadesTest, distanciasTest);
    tiemposMILP(m) = toc;
    costeMILP = calcularCoste(rutaMILP, distanciasTest);

    % --- Heurística del vecino más cercano ---
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
    costeNN = calcularCoste(rutaNN, distanciasTest);

    % --- Inferencia del MLP + decodificación voraz ---
    X_test = reshape(distanciasTest / max(distanciasTest(:)), [1, 1, numCiudades^2, 1]);
    Y_pred_1D = predict(redMLP_TSP, X_test);
    matrizProbabilidades = reshape(Y_pred_1D, [numCiudades, numCiudades]);
    matrizProbabilidades = (matrizProbabilidades + matrizProbabilidades') / 2;
    matrizProbabilidades(1:numCiudades+1:end) = 0;

    rutaMLP = zeros(1, numCiudades);
    nodoActual = 1;
    visitadosMLP = false(1, numCiudades);
    rutaMLP(1) = nodoActual;
    visitadosMLP(nodoActual) = true;
    for i = 2:numCiudades
        probabilidadesActual = matrizProbabilidades(nodoActual, :);
        probabilidadesActual(visitadosMLP) = -inf;
        [~, nodoSiguiente] = max(probabilidadesActual);
        rutaMLP(i) = nodoSiguiente;
        nodoActual = nodoSiguiente;
        visitadosMLP(nodoActual) = true;
    end
    costeMLP = calcularCoste(rutaMLP, distanciasTest);

    % --- Gaps de este mapa ---
    gapsNN(m)  = ((costeNN  - costeMILP) / costeMILP) * 100;
    gapsMLP(m) = ((costeMLP - costeMILP) / costeMILP) * 100;

    fprintf('  Mapa %d/%d listo (MILP: %.2fs)\n', m, numMapasTest, tiemposMILP(m));
end

tiempoTotal = toc(inicioTotal);
fprintf('\nEvaluación completa en %.1f s (%.1f s/mapa de media).\n', tiempoTotal, tiempoTotal/numMapasTest);

fprintf('\n=== RESULTADOS SOBRE %d MAPAS DE %d CIUDADES ===\n', numMapasTest, numCiudades);
fprintf('Heurística (NN):  gap medio %.2f%%  mediana %.2f%%  máx %.2f%%\n', mean(gapsNN), median(gapsNN), max(gapsNN));
fprintf('MLP:              gap medio %.2f%%  mediana %.2f%%  máx %.2f%%\n', mean(gapsMLP), median(gapsMLP), max(gapsMLP));

% --- Tabla LaTeX con los agregados (sustituye a la Tabla 4.2 actual) ---
fprintf('\n%% --- CÓDIGO LATEX LISTO PARA COPIAR ---\n');
fprintf('\\begin{table}[htbp]\n');
fprintf('    \\centering\n');
fprintf('    \\caption{Rendimiento medio sobre %d grafos de prueba de %d ciudades (óptimo exacto vía MILP).}\n', numMapasTest, numCiudades);
fprintf('    \\label{tab:metricas_rendimiento}\n');
fprintf('    \\begin{tabular}{lccc}\n');
fprintf('        \\toprule\n');
fprintf('        \\textbf{Modelo / Algoritmo} & \\textbf{Gap medio (\\%%)} & \\textbf{Gap mediana (\\%%)} & \\textbf{Gap máximo (\\%%)} \\\\\n');
fprintf('        \\midrule\n');
fprintf('        Heurística (NN) & %.2f & %.2f & %.2f \\\\\n', mean(gapsNN), median(gapsNN), max(gapsNN));
fprintf('        Perceptrón Multicapa (MLP) & %.2f & %.2f & %.2f \\\\\n', mean(gapsMLP), median(gapsMLP), max(gapsMLP));
fprintf('        \\bottomrule\n');
fprintf('    \\end{tabular}\n');
fprintf('\\end{table}\n');

% Funciones auxiliares -- copiadas sin cambios de tu Módulo 3.3
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