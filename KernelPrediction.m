function Prediction  =KernelPrediction(KernelRegression,X)
        %This is a matrix where each index corresponds the Kernel between the training samples and the samples you world like a preddication for
        %each rowe  eorresponds to a training sample and each column corresponds to a sample that you would like a prediction for
        Kt = kernelmatrix(KernelRegression.Kernel,KernelRegression.TrainingSamples,X',KernelRegression.Dummy(1),KernelRegression.Dummy(2),KernelRegression.Dummy(3));
        Prediction =Kt'*KernelRegression.Train;
end