% =========================================================================
% MÓDULO 3.2: ENTRENAMIENTO DE LA RED NEURONAL (MLP)
% =========================================================================
clc; clear; close all;

% 1. Cargar los datos generados previamente
disp('Cargando base de datos de entrenamiento...');
if ~isfile('DatosEntrenamientoTSP.mat')
    error('No se encuentra el archivo. Ejecuta primero GenerarDatosEntrenamiento.m');
end
load('DatosEntrenamientoTSP.mat', 'datosX', 'datosY', 'numCiudades');

% Convertir las celdas a matrices 4D 
X_train = zeros(1, 1, numCiudades^2, length(datosX));
Y_train = zeros(1, 1, numCiudades^2, length(datosY));
for i = 1:length(datosX)
    X_train(1,1,:,i) = reshape(datosX{i}, [numCiudades^2, 1]);
    Y_train(1,1,:,i) = reshape(datosY{i}, [numCiudades^2, 1]);
end

% 2. Diseño de la Arquitectura (Perceptrón Multicapa)
disp('Diseñando la arquitectura neuronal plana (MLP)...');
capas = [
    imageInputLayer([1 1 numCiudades^2], 'Normalization', 'none', 'Name', 'Entrada')
    
    fullyConnectedLayer(512, 'Name', 'FC_1')
    reluLayer('Name', 'ReLU_1')
    
    fullyConnectedLayer(1024, 'Name', 'FC_2')
    reluLayer('Name', 'ReLU_2')
    
    fullyConnectedLayer(512, 'Name', 'FC_3')
    reluLayer('Name', 'ReLU_3')
    
    fullyConnectedLayer(numCiudades^2, 'Name', 'Salida_Probabilidades')
    sigmoidLayer('Name', 'Sigmoide')
    regressionLayer('Name', 'Perdida')
];

% 3. Opciones de Entrenamiento
disp('Configurando opciones de entrenamiento (ADAM)...');
opciones = trainingOptions('adam', ...
    'MaxEpochs', 250, ...               
    'MiniBatchSize', 128, ...           
    'InitialLearnRate', 0.001, ...     
    'Plots', 'training-progress', ...  
    'Verbose', false, ...
    'ExecutionEnvironment', 'cpu');    

% 4. Entrenar la Red
disp('Iniciando entrenamiento...');
tic;
redMLP_TSP = trainNetwork(X_train, Y_train, capas, opciones);
tiempoEntrenamiento = toc;
fprintf('\n¡Entrenamiento completado en %.2f segundos!\n', tiempoEntrenamiento);

% 5. Guardar la Red Entrenada
save('RedMLP_TSP_Entrenada.mat', 'redMLP_TSP', 'numCiudades');
disp('Red guardada. Lista para inferencia.');

% =========================================================================
% MÓDULO VISUAL: EVIDENCIA DEL COLAPSO TOPOLÓGICO (LaTeX Export)
% =========================================================================
disp('Generando figura probatoria de inferencia (Contraste corregido)...');

% Extracción y reconstrucción del mapa base
muestra_idx = 1; 
X_test = X_train(1,1,:,muestra_idx);
Y_real = reshape(Y_train(1,1,:,muestra_idx), [numCiudades, numCiudades]);

% Inferencia ciega del MLP
Y_pred_vector = predict(redMLP_TSP, X_test);
Y_pred = reshape(Y_pred_vector, [numCiudades, numCiudades]);

fig = figure('Position', [100, 100, 800, 400], 'Color', 'w');

% Panel 1: Verdad Fundamental (Óptimo)
subplot(1, 2, 1);
imagesc(Y_real);
colormap(gca, flipud(gray)); 
cb1 = colorbar; cb1.Color = 'k'; cb1.TickLabelInterpreter = 'tex';
title('\bf A. Matriz Objetivo (MILP)', 'Interpreter', 'tex', 'FontSize', 12, 'Color', 'k');
axis square; 
set(gca, 'TickLabelInterpreter', 'tex', 'FontSize', 10, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 0.8);
xlabel('Índice del Nodo', 'Interpreter', 'tex', 'Color', 'k'); 
ylabel('Índice del Nodo', 'Interpreter', 'tex', 'Color', 'k');

% Panel 2: Predicción Difusa (Fracaso)
subplot(1, 2, 2);
imagesc(Y_pred);
colormap(gca, parula); 
cb2 = colorbar; cb2.Color = 'k'; cb2.TickLabelInterpreter = 'tex';
title('\bf B. Predicción del MLP', 'Interpreter', 'tex', 'FontSize', 12, 'Color', 'k');
axis square; 
set(gca, 'TickLabelInterpreter', 'tex', 'FontSize', 10, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 0.8);
xlabel('Índice del Nodo', 'Interpreter', 'tex', 'Color', 'k');

% Exportación vectorial
exportgraphics(fig, 'colapso_inferencia_mlp.pdf', 'ContentType', 'vector');
disp('Gráfico guardado como "colapso_inferencia_mlp.pdf".');