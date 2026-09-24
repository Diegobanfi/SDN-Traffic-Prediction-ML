%sigmas  = logspace(-2,2,9);
%lambdas = logspace(-8,-1,8);

%for s = sigmas
  %for lam = lambdas
        %kmdl = fitrkernel(X_train, Y_train, 'Standardize', true,'KernelScale', s,'Lambda',lam,'KFold', 5);   
        %lossCV = kfoldLoss(kmdl);  % MSE medio sui fold

    %fprintf('sigma=%9.3g  lambda=%9.3g  CVloss=%12.6g\n', s, lam, lossCV);
  %end
%end

% predizione su test 
%Yhat3 = predict(kmdl, X_test);  % predizione su test [web:40]
%NRSME3 = sqrt(sum((Yhat3-Y_test).^2)/length(Y_test))/mean(Y_test); 
%disp(['RBF NRMSE  ', num2str(NRSME3*100),'[%]'])


sigma = 10;
lambda = 1e-08;

mdl = fitrkernel(X_train, Y_train, ...
    'Standardize', true, ...
    'KernelScale', sigma, ...
    'Lambda', lambda);

Yhat3 = predict(mdl, X_test);
NRMSE3 = sqrt(mean((Yhat3 - Y_test).^2)) / mean(Y_test) * 100;
fprintf('TEST: sigma=%g lambda=%g  NRMSE=%.2f%%\n', sigma, lambda, NRMSE3);