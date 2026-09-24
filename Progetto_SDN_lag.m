% ML PROJECT SDN TRAFFIC REGRESSION
% DIEGO BANFI 306957
% OBB: PREVEDERE I PACCHETTI IN A 6 ISTANTI SUCCESSIVI(DOPO 30 MINUTI)
% METODI: LAGMATRIX, LS, SGD, KERNEL METHODS, SVR, RT , RF

close all 
clear
clc
format short % for better visualization 

%read .csv 
train_data = readtable('SONICATEL traffic train.csv');
test_data = readtable('SONICATEL traffic test.csv');

%% Scelta horizon e target
h = 3;            % h=3

%target
IN_train = train_data.IN;  % Vettore pacchetti IN, RESPONSE

%target da prevedere ( dopo 60 minuti)
Y_train = IN_train(h+1:end);

%rimuovo le ultime 12 osservazioni delle feature del train e test xke non ho il target corrispondente

%train
X_train = train_data(1:end-h,4 :end);

%test
IN_test = test_data.IN;
Y_test  = IN_test(h+1:end);
X_test  = test_data(1:end-h, 4:end);

%lag
dx = 0; %3 ottimale, RT PIU BASSO
disp(['LAG:', num2str(dx)])

tempX_train = lagmatrix(X_train,0:1:dx);

% Elimino le righe iniziali corrispondenti al lag per sincronizzare i dati
tempX_train = tempX_train(dx+1:end, :);
tempY_train = Y_train(dx+1:end, :);

%AGGIORNO X_TRAIN E Y_TRAIN
X_train=tempX_train;
Y_train=tempY_train;

%dimensioni
[N, NX] = size(X_train);

%test lag
tempX_test = lagmatrix(X_test,0:1:dx);

tempX_test = tempX_test(dx+1:end, :);
tempY_test = Y_test(dx+1:end, :);
    
% Aggiorno le matrici di test
X_test=tempX_test;
Y_test=tempY_test;

%dimensioni
[N_test, NX_test] = size(X_test);
disp(['DIMENSIONI DATA_TEST:', num2str([N NX]) ])

%converto in array le tabelle (matrici numeriche) compatibile per i vari metodi
X_train = table2array(X_train);   % converte la table in double
X_test  = table2array(X_test);



%% 1-) Least square - BATCH
M_hat0= X_train\Y_train; %pseudo inversa
Yhat0= X_test*M_hat0;
%errore
NRSME0 = sqrt(sum((Yhat0-Y_test).^2)/length(Y_test))/mean(Y_test); 
disp(['LS NRMSE  ', num2str(NRSME0*100),'[%]'])


%% 2-) LMS algorithm : SGD

%Aggiungo bias in modo tale che il mio modello non è vincolato a passare per l'origine
%Xtr = [ones(N,1), X_train];
%Xte = [ones(N,1), X_test];

% Iperparametri SGD
Max_Iter = 1000;     % numero di passate sul dataset
eta = 1e-18;      % learning rate 

% Vettore pesi e vettore errore
M_hat1 = zeros(NX,Max_Iter); %colonne=dim_feauture
err = zeros(1,Max_Iter);

for i = 1:Max_Iter-1
    idx = randi([1, N]);         % random (SGD)
    %idx = mod(i-1, N) + 1;

    % predizione su quel campione: yhat = X(idx,:)*w
    % errore (scalare): e = Y(idx) - X(idx,:)*w
    M_hat1(:, i+1) = M_hat1(:, i) + eta * X_train(idx,:)' * ( Y_train(idx) - X_train(idx,:) * M_hat1(:, i) );

    err(i) = abs( Y_train(idx) - X_train(idx,:) * M_hat1(:, i) );   % errore assoluto
end

w_finale = M_hat1(:, end); %scelgo ultima colonna, l'ultima aggiornata
Yhat1 = X_test * w_finale;

NRMSE1 = sqrt(mean((Yhat1 - Y_test).^2)) / mean(Y_test) * 100;
fprintf('LMS/SGD NRMSE = %.2f %%\n', NRMSE1);



%% 3-) KERNEL REGRESSION RBF

%aggiungere ciclo for in cui faccio vedere come ho scelto sigma e lambda

%RBF
sigma = 10;
lambda = 1e-08;

k_rbf = fitrkernel(X_train, Y_train, 'Standardize', true, 'KernelScale', sigma, 'Lambda', lambda);

Yhat2 = predict(k_rbf, X_test);
NRMSE2 = sqrt(mean((Yhat2 - Y_test).^2)) / mean(Y_test) * 100;


disp(['KERNEL REGRESSION RBF NRMSE = ', num2str(NRMSE2), sigma, lambda, ' [%]'])

%% 3.1-) Polynomial Kernel
ker = 'poly'; %definisco kernel

%caso1
b = 1; %bias of the linear/polynomial
d = 10; %dregree of linear/polynomial kernel
param = [b, d]; %contiene parametri kernel che sto usando
Reg = 0.8; %termine di regolarizzazione

%Calcolo K(simmetrica) e 'a'
KernelRegression1 = KernelRidgeRegression(ker,X_train,param,Y_train,Reg); 

%prediction tramite polinomial kernel
Yhat7 = KernelPrediction(KernelRegression1,X_test); 

NRMSE7 = sqrt(mean((Yhat7 - Y_test).^2)) / mean(Y_test) * 100;


disp(['KERNEL REGRESSION POLY NRMSE = ', num2str(NRMSE7), sigma, lambda, ' [%]']);
%% 5-) RT

tree1 = fitrtree(X_train,Y_train);
Yhat4 = predict(tree1,X_test);

%view(tree1,'Mode','graph');
NRSME4 = sqrt(sum((Yhat4-Y_test).^2)/length(Y_test))/mean(Y_test); 
disp(['RT NRMSE  ', num2str(NRSME4*100),'[%]'])


%minLeaf_list = [1 5 10 20 50 100];
%for m = minLeaf_list
    %tree_tmp = fitrtree(X_train, Y_train, 'MinLeafSize', m, 'MaxNumSplits', 500);
    %Yhat_tmp = predict(tree_tmp, X_test);
    %NRMSE_tmp = sqrt(sum((Yhat_tmp-Y_test).^2)/length(Y_test))/mean(Y_test);
    %fprintf('MinLeafSize = %3d → NRMSE = %.2f [%%]\n', m, NRMSE_tmp*100);
%end

%% 6-) RF
numTrees = [1 5 10 50 100 200];
nF = numel(numTrees); %n_el array
YhatRF_matrice = zeros(length(Y_test), nF);   % ogni colonna = una Yhat5 di RF

for i = 1:nF
    numtrees = numTrees(i);
    forest1 = TreeBagger(numtrees,X_train,Y_train,'Method','regression'); %treebagger per generare random forest
    Yhat5 = predict(forest1,X_test); %predizione foresta
    YhatRF_matrice(:,i) = Yhat5;
    NRSME5 = sqrt(sum((Yhat5-Y_test).^2)/length(Y_test))/mean(Y_test);
    disp(['RF NRMSE  ', num2str(numtrees), ' = ', num2str(NRSME5*100),'[%]'])
end


%% 4-)SVM REGRESSION 

%RBF KERNEL
svm_rbf = fitrsvm(X_train, Y_train,'KernelFunction','rbf','Standardize',true);% Standardize standardizza automaticamente le feature di input prima di addestrare la SVR

%Numeri di SV che trovo nel train e uso nel testing
numSV = sum(svm_rbf.IsSupportVector);
fprintf('Num Support Vectors: %d / %d (%.1f%%)\n', numSV, size(X_train,1), 100*numSV/size(X_train,1));


%Predizione SVM RBF
Yhat3 = predict(svm_rbf, X_test);

NRMSE3 = sqrt(mean((Yhat3 - Y_test).^2)) / mean(Y_test) * 100;

disp(['SVM RBF NRMSE = ', num2str(NRMSE3),' [%]']);

%POLINOMIAL KERNEL
svm_poly = fitrsvm(X_train,Y_train,'KernelFunction','polynomial','PolynomialOrder',3); %'PolynomialOrder' indica che grado del polinomio voglio scegliere

%Numeri di SV che trovo nel train e uso nel testing
numSV = sum(svm_poly.IsSupportVector);
fprintf('Num Support Vectors: %d / %d (%.1f%%)\n', numSV, size(X_train,1), 100*numSV/size(X_train,1));


%PREDIZIONE SVM POLINOMIAL
Yhat6 = predict(svm_poly,X_test);

NRMSE6 = sqrt(mean((Yhat6 - Y_test).^2)) / mean(Y_test) * 100;

disp(['SVM POLINOMIAL NRMSE = ', num2str(NRMSE6),' [%]']);

%% PLOT T_TEST VS PREDIZIONE DI RT E RF
t = 1:length(Y_test);
figure; hold on; grid on;
plot(t, Y_test, 'k', 'LineWidth', 2);
plot(t, Yhat4,  'g', 'LineWidth', 1.2);
plot(t, YhatRF_matrice, 'LineWidth', 1);  % plottando una matrice, MATLAB disegna tutte le colonne
labels = cell(1, 2+nF);
labels{1} = 'Dati reali';
labels{2} = 'RT (albero singolo)';
for i = 1:nF
    labels{2+i} = sprintf('RF (%d alberi)', numTrees(i));  % label dinamica 
end
legend(labels, 'Location','best');  % legenda con lista di label 
xlabel('Tempo (campioni)');
ylabel('Pacchetti (IN+12)');
title('Confronto: RT vs tutte le RF');
hold off;


%% PLOT T_TEST VS PREDIZIONE DI TUTTI I METODI
t = 1:length(Y_test);

figure; hold on; grid on;

hReal = plot(t, Y_test, 'k', 'LineWidth', 2);
h0 = plot(t, Yhat0, 'b', 'LineWidth', 1);
h1 = plot(t, Yhat1, 'm', 'LineWidth', 1);
h2 = plot(t, Yhat2, 'Color', [0.2 0.7 0.9], 'LineWidth', 1);
h3 = plot(t, Yhat3, 'Color', [0.9 0.5 0.1], 'LineWidth', 1);
h4 = plot(t, Yhat4, 'g', 'LineWidth', 1);
h5 = plot(t, Yhat5, 'r', 'LineWidth', 1);

legend([hReal h0 h1 h2 h3 h4 h5],{'Dati reali','LS','LMS/SGD','Kernel RBF','SVR RBF','RT','RF (200 alberi)'},'Location','best');   

xlabel('Tempo (campioni)');
ylabel('Pacchetti (IN+6)');
title('Confronto modelli: predizioni vs Y\_test');
hold off;


%% NN

