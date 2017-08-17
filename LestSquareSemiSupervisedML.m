function Z = LestSquareSemiSupervisedML( X, Y, L, K, d, beta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Least-squares Method for Semi-supervised manifold learning
% X: samples
% Y: latent variable of labeled samples
% L: X(:,1:L) labeled samples
% K: the number of neighbors
% d: low-dimension of latent space
% 0<alpha1, alpha2<1, lambda: parameters
%
% Reference:
% Yang, Xin, et al. 
% "Semi-supervised nonlinear dimensionality reduction." 
% Proceedings of the 23rd international conference on Machine learning. 
% ACM, 2006.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(X,2);
[Nei, IndL, IndP, IndN] = NeighborSelect(X, L, K);

% create Phi for all the data set
BI = cell(N,1);
for i=1:N
    Xi = X(:,Nei(:,i));
    Xi = Xi - repmat( mean(Xi,2), [1,K] );
    W = Xi'*Xi; W = (W+W')/2;
    [Vi,Si] = schur(W);
    [~,Ji] = sort(-diag(Si)); 
    Vi = Vi(:,Ji(1:d));  

    % construct Gi
    Gi = [repmat(1/sqrt(K),[K,1]) Vi];  
    % compute the local orthogonal projection Bi = I-Gi*Gi' 
    % that has the null space span([e,Theta_i^T]). 
    BI{i} = eye(K)-Gi*Gi';  
end

B = zeros(N);
for i=1:N
    Ii = Nei(:,i)';
    B(Ii,Ii) = B(Ii,Ii)+BI{i};
end;
B = (B+B')/2;

if beta==0
    M22=B(L+1:end,L+1:end);
    M12=B(1:L,L+1:end);
    Y2=M22\(M12'*Y');
    Z=[Y,Y2'];
else
    Tmp=zeros(N);
    Tmp(1:L,1:L)=beta*eye(L);
    B=B+Tmp;
    b=[beta.*Y';zeros(N-L,d)];
    Z=(B\b)';
end
    
