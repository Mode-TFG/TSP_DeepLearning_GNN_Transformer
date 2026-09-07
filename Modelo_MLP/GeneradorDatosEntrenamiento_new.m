% =========================================================================
% MÓDULO 3.1: GENERACIÓN DE DATOS PARA LA IA (ETIQUETADO SUPERVISADO)
% =========================================================================
clc; clear; close all;

numMapas = 5000;      % Reducido a 5000 para una demostración ágil
numCiudades = 20;     % Mapas pequeños para entrenar rápido en el portátil

datosX = cell(numMapas, 1);
datosY = cell(numMapas, 1);
datosCiudades = cell(numMapas, 1); % [CORRECCIÓN]: Guardar coordenadas espaciales

disp('Generando base de datos de entrenamiento. Esto puede tardar un poco...');
tic;
for m = 1:numMapas
    % 1. Generar mapa
    ciudades = rand(numCiudades, 2);
    distancias = pdist2(ciudades, ciudades);
    
    % 2. Resolver con el Óptimo Matemático (MILP)
    rutaOptima = resolverTSPExacto(ciudades, distancias);
    
    % 3. Crear la matriz objetivo (Label) para la red neuronal
    Y_optima = zeros(numCiudades, numCiudades);
    for i = 1:(length(rutaOptima)-1)
        Y_optima(rutaOptima(i), rutaOptima(i+1)) = 1;
        Y_optima(rutaOptima(i+1), rutaOptima(i)) = 1; % TSP Simétrico
    end
    
    %Cierre del Ciclo Hamiltoniano
    Y_optima(rutaOptima(end), rutaOptima(1)) = 1;
    Y_optima(rutaOptima(1), rutaOptima(end)) = 1;

    % 4. Guardar datos
    datosX{m} = distancias / max(distancias(:)); % Matriz normalizada
    datosY{m} = Y_optima;
    datosCiudades{m} = ciudades;                 % Coordenadas retenidas
    
    if mod(m, 100) == 0
        fprintf('Completados %d / %d mapas...\n', m, numMapas);
    end
end
tiempoGen = toc;
fprintf('Base de datos generada en %.2f segundos.\n', tiempoGen);

% Guardar dataset enriquecido
save('DatosEntrenamientoTSP.mat', 'datosX', 'datosY', 'datosCiudades', 'numCiudades');
disp('Datos guardados en DatosEntrenamientoTSP.mat');

% =========================================================================
% VISUALIZACIÓN DE ALTA CALIDAD: ANATOMÍA DE UNA MUESTRA (LaTeX Export)
% =========================================================================
disp('Generando figura representativa del conjunto de datos...');

% Extraemos la muestra número 1 para ilustrar el paradigma
muestra_idx = 1;
X_sample = datosX{muestra_idx};
Y_sample = datosY{muestra_idx};
C_sample = datosCiudades{muestra_idx};

fig = figure('Position', [100, 100, 1200, 350], 'Color', 'w');

% --- Panel 1: Topología del Problema (Grafo) ---
subplot(1, 3, 1);
hold on; grid on;
% Reconstruir la ruta para dibujarla
[r, c] = find(triu(Y_sample) == 1);
for i = 1:length(r)
    plot([C_sample(r(i),1), C_sample(c(i),1)], [C_sample(r(i),2), C_sample(c(i),2)], '-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 1.5);
end
scatter(C_sample(:,1), C_sample(:,2), 45, 'MarkerFaceColor', [0.9290 0.6940 0.1250], 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
title('\bf A. Espacio Euclidiano y Ruta Óptima', 'Interpreter', 'tex', 'FontSize', 12, 'Color', 'k');
axis equal; axis([0 1 0 1]); box on;
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10, 'XColor', 'k', 'YColor', 'k');

% --- Panel 2: Entrada de la Red (Matriz de Características X) ---
subplot(1, 3, 2);
imagesc(X_sample); 
colormap(gca, 'parula'); % Colormap técnico para distancias
cb1 = colorbar; 
cb1.TickLabelInterpreter = 'latex'; cb1.Color = 'k';
title('\bf B. Tensor de Entrada (Matriz $\mathbf{X}$)', 'Interpreter', 'latex', 'FontSize', 12, 'Color', 'k');
subtitle('Distancias relativas normalizadas', 'Interpreter', 'latex', 'FontSize', 10, 'Color', 'k');
axis square; 
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10, 'XColor', 'k', 'YColor', 'k');
xlabel('Índice del Nodo', 'Interpreter', 'tex'); 
ylabel('Índice del Nodo', 'Interpreter', 'tex');

% --- Panel 3: Etiqueta Supervisada (Matriz Objetivo Y) ---
subplot(1, 3, 3);
imagesc(Y_sample);
colormap(gca, flipud(gray)); % Blanco = 0, Negro = 1 para máximo contraste
cb2 = colorbar; 
cb2.Ticks = [0, 1]; cb2.TickLabelInterpreter = 'latex'; cb2.Color = 'k';
title('\bf C. Matriz Objetivo (Label $\mathbf{Y}$)', 'Interpreter', 'latex', 'FontSize', 12, 'Color', 'k');
subtitle('Adyacencia de la ruta MILP', 'Interpreter', 'latex', 'FontSize', 10, 'Color', 'k');
axis square; 
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 10, 'XColor', 'k', 'YColor', 'k');
xlabel('Índice del Nodo', 'Interpreter', 'tex');

% Exportar la figura en alta resolución
exportgraphics(fig, 'muestra_entrenamiento_IA.pdf', 'ContentType', 'vector');
disp('Gráfico guardado como "muestra_entrenamiento_IA.pdf". Listo para importar en LaTeX.');

% =========================================================================
% FUNCIÓN AUXILIAR: Solucionador Exacto (MILP)
% =========================================================================
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
        Aeq(viajes(k,1), k) = 1;
        Aeq(viajes(k,2), k) = 1;
    end
    beq = 2 * ones(n, 1);
    
    intcon = 1:numViajes;
    lb = zeros(numViajes, 1);
    ub = ones(numViajes, 1);
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
                nuevaA = zeros(1, numViajes);
                nuevaA(idxSubtour) = 1;
                Aineq = [Aineq; nuevaA];
                bineq = [bineq; length(nodosSubtour) - 1];
            end
        end
    end
    
    ruta = zeros(1, n);
    nodoActual = 1;
    ruta(1) = nodoActual;
    for i = 2:n
        vecinos = neighbors(G, nodoActual);
        if i == 2
            nodoSiguiente = vecinos(1);
        else
            nodoSiguiente = vecinos(vecinos ~= ruta(i-2));
        end
        ruta(i) = nodoSiguiente;
        nodoActual = nodoSiguiente;
    end
end