% =========================================================================
% MÓDULO 1: ENTORNO TSP Y EXTRACCIÓN DE EVIDENCIA (TFG)
% =========================================================================
clc; clear; close all;

% 1. Parámetros del problema
numCiudades = 40; 

% 2. Generar mapa aleatorio (Coordenadas X, Y entre 0 y 1)
rng(42); % Semilla fija para reproducibilidad
ciudades = rand(numCiudades, 2);
distancias = pdist2(ciudades, ciudades);

% =========================================================================
% LÍNEA BASE CERO: Heurística del Vecino Más Cercano (Nearest Neighbor)
% =========================================================================
disp('Calculando aproximación heurística...');
tic; 
rutaNN = zeros(1, numCiudades);
nodoActual = 1; 
visitados = false(1, numCiudades);
rutaNN(1) = nodoActual;
visitados(nodoActual) = true;

for i = 2:numCiudades
    distanciasDesdeActual = distancias(nodoActual, :);
    distanciasDesdeActual(visitados) = inf;
    [~, nodoSiguiente] = min(distanciasDesdeActual);
    
    rutaNN(i) = nodoSiguiente;
    nodoActual = nodoSiguiente;
    visitados(nodoActual) = true;
end
tiempoNN = toc; 
costeNN = calcularCosteRuta(rutaNN, distancias);

% =========================================================================
% LÍNEA BASE EXACTA: Programación Lineal Entera Mixta (MILP)
% =========================================================================
disp('Calculando solución exacta matemática (intlinprog)...');
viajes = nchoosek(1:numCiudades, 2);
numViajes = size(viajes, 1);
distanciasViajes = zeros(numViajes, 1);

for k = 1:numViajes
    distanciasViajes(k) = distancias(viajes(k,1), viajes(k,2));
end

Aeq = spalloc(numCiudades, numViajes, numCiudades*(numCiudades-1));
for k = 1:numViajes
    Aeq(viajes(k,1), k) = 1;
    Aeq(viajes(k,2), k) = 1;
end
beq = 2 * ones(numCiudades, 1);
intcon = 1:numViajes;
lb = zeros(numViajes, 1);
ub = ones(numViajes, 1);
opciones = optimoptions('intlinprog', 'Display', 'off');

Aineq = [];
bineq = [];
numSubtours = 2; 
tic; 

while numSubtours > 1
    [x_opt, costeExacto] = intlinprog(distanciasViajes, intcon, Aineq, bineq, Aeq, beq, lb, ub, opciones);
    aristasSeleccionadas = find(round(x_opt) == 1);
    
    G = graph(viajes(aristasSeleccionadas, 1), viajes(aristasSeleccionadas, 2));
    [bins, ~] = conncomp(G);
    numSubtours = max(bins);
    
    if numSubtours > 1
        for k = 1:numSubtours
            nodosSubtour = find(bins == k);
            idxSubtour = ismember(viajes(:,1), nodosSubtour) & ismember(viajes(:,2), nodosSubtour);
            
            nuevaA = zeros(1, numViajes);
            nuevaA(idxSubtour) = 1;
            Aineq = [Aineq; nuevaA];
            bineq = [bineq; length(nodosSubtour) - 1];
        end
    end
end
tiempoExacto = toc;

rutaExacta = zeros(1, numCiudades);
nodoActual = 1;
rutaExacta(1) = nodoActual;
for i = 2:numCiudades
    vecinos = neighbors(G, nodoActual);
    if i == 2
        nodoSiguiente = vecinos(1);
    else
        nodoSiguiente = vecinos(vecinos ~= rutaExacta(i-2));
    end
    rutaExacta(i) = nodoSiguiente;
    nodoActual = nodoSiguiente;
end
brechaOpt = ((costeNN - costeExacto)/costeExacto)*100;

% =========================================================================
% VISUALIZACIÓN CON CALIDAD DE PUBLICACIÓN (Exportable a LaTeX)
% =========================================================================
fig = figure('Position', [100, 100, 900, 450], 'Color', 'w');

% --- Subplot 1: Heurística ---
subplot(1,2,1);
plot(ciudades(rutaNN, 1), ciudades(rutaNN, 2), '-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.2); hold on;
plot([ciudades(rutaNN(end), 1), ciudades(rutaNN(1), 1)], [ciudades(rutaNN(end), 2), ciudades(rutaNN(1), 2)], '-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.2);

% Modificación: Reducción del área nodal a 20 para evitar oclusión de aristas cortas.
% Eliminación del marcador jerárquico inicial para asegurar simetría espacial.
scatter(ciudades(:,1), ciudades(:,2), 20, 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerEdgeColor', 'k', 'LineWidth', 0.8); 

title('\bf A. Heurística (Vecino Más Cercano)', 'Interpreter', 'tex', 'FontSize', 13, 'Color', 'k');
subtitle(sprintf('Coste: %.2f', costeNN), 'Interpreter', 'latex', 'FontSize', 11, 'Color', 'k');
axis equal; axis([0 1 0 1]); box on;
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 0.8);

% --- Subplot 2: Exacto ---
subplot(1,2,2);
plot(ciudades(rutaExacta, 1), ciudades(rutaExacta, 2), '-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 1.2); hold on;
plot([ciudades(rutaExacta(end), 1), ciudades(rutaExacta(1), 1)], [ciudades(rutaExacta(end), 2), ciudades(rutaExacta(1), 2)], '-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 1.2);

% Modificación: Aplicación de la misma corrección isométrica al oráculo matemático.
scatter(ciudades(:,1), ciudades(:,2), 20, 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);

title('\bf B. Óptimo Matemático (MILP)', 'Interpreter', 'tex', 'FontSize', 13, 'Color', 'k');
subtitle(sprintf('Coste: %.2f', costeExacto), 'Interpreter', 'latex', 'FontSize', 11, 'Color', 'k');
axis equal; axis([0 1 0 1]); box on;
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 11, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 0.8);

exportgraphics(fig, 'comparativa_rutas.pdf', 'ContentType', 'vector');
disp('Gráfico guardado como "comparativa_rutas.pdf". Nodos normalizados y simetría restaurada.');

% =========================================================================
% GENERACIÓN AUTOMÁTICA DEL CÓDIGO LATEX PARA LA TABLA
% =========================================================================
fprintf('\n\n%% --- CÓDIGO LATEX LISTO PARA COPIAR ---\n');
fprintf('\\begin{table}[htbp]\n');
fprintf('    \\centering\n');
fprintf('    \\caption{Comparativa de rendimiento en un grafo de %d ciudades.}\n', numCiudades);
fprintf('    \\label{tab:rendimiento_baseline}\n');
fprintf('    \\begin{tabular}{lccc}\n');
fprintf('        \\toprule\n');
fprintf('        \\textbf{Algoritmo} & \\textbf{Tiempo (s)} & \\textbf{Coste Total} & \\textbf{Brecha de Optimalidad} \\\\\n');
fprintf('        \\midrule\n');
fprintf('        Óptimo (MILP) & %.4f & %.2f & 0.00\\%% \\\\\n', tiempoExacto, costeExacto);
fprintf('        Heurística (NN) & %.4f & %.2f & +%.2f\\%% \\\\\n', tiempoNN, costeNN, brechaOpt);
fprintf('        \\bottomrule\n');
fprintf('    \\end{tabular}\n');
fprintf('\\end{table}\n');
fprintf('%% --------------------------------------\n\n');

function costeTotal = calcularCosteRuta(ruta, matrizDistancias)
    costeTotal = 0;
    for k = 1:(length(ruta)-1)
        costeTotal = costeTotal + matrizDistancias(ruta(k), ruta(k+1));
    end
    costeTotal = costeTotal + matrizDistancias(ruta(end), ruta(1));
end