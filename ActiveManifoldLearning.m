function [Align, IndexU, IndexL, Time] = ActiveManifoldLearning( X, options, AL, L)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Active manifold learning
% Input
% X: samples
% options.L: the number of labels we can assign
% options.K: the number of neighbors
% options.d: low-dimension of latent space
% options.ML: manifold learning methods: LTSA, 
%                                        LLE,...
% options.AL: active learning methods: 
%                   RD (Random),
%                   GD (Geodesic Distance), 
%                   DPP (Determinantal Point Processes), 
%                   GC (Gerschgorin Circle),
%                   FGC (Fast Gerschgorin Circle),...
%
% Output
% Align: alignment matrix
% IndexU: the index of unlabeled samples
% IndexL: the index of labeled samples
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch options.ML
    case 'LTSA'
        % STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
        N = size(X,2);
        X2 = sum(X.^2,1);
        distance = repmat(X2,N,1)+repmat(X2',1,N)-2*(X'*X);
        [~,index] = sort(distance);
        Nei = index(1:options.K,:);
        NN = Nei(2:end,:);
        
        % STEP2: FIND LOCAL TANGENT SPACE
        BI = cell(N,1);
        for i=1:N
            Xi = X(:,Nei(:,i));
            Xi = Xi - repmat( mean(Xi,2), [1,options.K] );
            W = Xi'*Xi; W = (W+W')/2;
            [Vi,Si] = schur(W);
            [~,Ji] = sort(-diag(Si)); 
            Vi = Vi(:,Ji(1:options.d));  

            % construct Gi
            Gi = [repmat(1/sqrt(options.K),[options.K,1]) Vi];  
            % compute the local orthogonal projection Bi = I-Gi*Gi' 
            % that has the null space span([e,Theta_i^T]). 
            BI{i} = eye(options.K)-Gi*Gi';  
        end

        % STEP3: Compute aligment matrix
        B = zeros(N);%speye(N);
        for i=1:N
            Ii = Nei(:,i);
            B(Ii,Ii) = B(Ii,Ii)+BI{i};
        end;
        Align = (B+B')/2;
        
    
    case 'LLE'
        % STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
        [D,N] = size(X);
        X2 = sum(X.^2,1);
        distance = repmat(X2,N,1)+repmat(X2',1,N)-2*(X'*X);
        [~,index] = sort(distance);
        neighborhood = index(2:(1+options.K),:);
        
        NN = neighborhood;
        
        % STEP2: SOLVE FOR RECONSTRUCTION WEIGHTS
        if(options.K>D) 
          tol=1e-3; % regularlizer in case constrained fits are ill conditioned
        else
          tol=0;
        end
        W = zeros(options.K,N);
        for ii=1:N
           z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,options.K); % shift ith pt to origin
           C = z'*z;                                        % local covariance
           C = C + eye(options.K)*tol*trace(C);                   % regularlization (K>D)
           W(:,ii) = C\ones(options.K,1);                           % solve Cw=1
           W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
        end
        % STEP 3: COMPUTE EMBEDDING FROM EIGENVECTS OF COST MATRIX M=(I-W)'(I-W)
        Align = eye(N);%sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
        for ii=1:N
           w = W(:,ii);
           jj = neighborhood(:,ii);
           Align(ii,jj) = Align(ii,jj) - w';
           Align(jj,ii) = Align(jj,ii) - w;
           Align(jj,jj) = Align(jj,jj) + w*w';
        end
        
%     case 'Laplacian'
%         n = size(X,2);
%         N=n;
%         A = sparse(n,n);
%         step = 100;  
%         for i1=1:step:n    
%             i2 = i1+step-1;
%             if (i2> n) 
%                 i2=n;
%             end
%             XX= X(:,i1:i2);  
%             dt = L2_distance(XX,X,0);
%             [Z,I] = sort ( dt,2);
%             for i=i1:i2
%                 for j=2:options.K+1
%                     A(i,I(i-i1+1,j))= Z(i-i1+1,j); 
%                     A(I(i-i1+1,j),i)= Z(i-i1+1,j); 
%                 end    
%             end
%         end
%         W = A;
%         [A_i, A_j, A_v] = find(A);  % disassemble the sparse matrix
%         for i = 1: size(A_i)  
%             W(A_i(i), A_j(i)) = 1;
%         end;
%         D = sum(W(:,:),2);   
%         Align = full(spdiags(D,0,speye(size(W,1)))-W);


end




switch AL
    case 'RD'
        tic;
        Index = randperm(N);
        IndexL = Index(1:L);
        IndexU = Index(1+L:end);
        Time=toc;
    case 'GD'
        tic;
        [IndexU, IndexL] = ActiveLearningGD( X, options, L );
        Time=2*toc;
    case 'DPP'
        tic;
        [IndexU, IndexL] = ActiveLearningDPP( X, options, L );
        Time=toc;
    case 'GC'
        tic;
        [IndexU, IndexL] = ActiveLearningGC( Align, L );
        Time=toc;
    case 'HGC'
        tic;
        [IndexU, IndexL] = ActiveLearningHGC( Align, L );
        Time=toc;
    case 'FGC'
        tic;
        [IndexU, IndexL] = ActiveLearningFFGC( Align, L );
        Time=toc;
end
