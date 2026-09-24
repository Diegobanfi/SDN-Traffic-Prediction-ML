function KernelRegression = KernelRidgeRegression(ker,X,parameters,Target,RegulationTerm)
    %       ker: 'poly','rbf'
    %       X: data matrix with training samples in rows and features in columns

    %parameters:
          %If RBF kernel or sam kernel must be one value 
          %sigma: width of the RBF kernel

         % for polynomial kernel 'poly' the paramters are in the
         % form  parameters=[b d]
         %       b:     bias in the linear and polinomial kernel
         %       d:     degree in the polynomial kernel

         %Linear kernels 'lin' parameters=b;
         %       b:     bias in the linear and polinomial kernel


    %Target:row vector of continuous values to be predicted

    % %K:Gram matrix, each element corresponds to the kernel of two feature vectors


    if (strcmp(ker,'rbf'))

        if  length(parameters)~=1
           error('Error: RBF kernel and Sam kennel only needs one parameter') 
        end

        b=1;
        d=1;
        sigma=parameters;
        K = kernelmatrix(ker,X',X',sigma,b,d);

    end

    if strcmp(ker,'poly')
    
        if  length(parameters)~=2
            error('Error: polynomial kernels need two parameters') 
        end

        sigma=1;
        b=parameters(1);
        d=parameters(2);
        K = kernelmatrix(ker,X',X',sigma,b,d);

    end
    



    %Regulation matrix
    RegularizationMatrix=eye(length(K(:,1)));

    %Name of kernel
    KernelRegression.Kernel=ker;
    %store Parameters
    KernelRegression.Parameters=parameters;
    %Store training parameters
    KernelRegression.TrainingSamples=X';
    %This is the vector of  parameters (K+labda I)^-1 Targets
    KernelRegression.Train=((K+RegulationTerm*RegularizationMatrix)^-1)*Target;

    KernelRegression.Dummy=[sigma,b,d];



end    