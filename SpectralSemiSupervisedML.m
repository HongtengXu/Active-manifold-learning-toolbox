function Z = SpectralSemiSupervisedML( X, Y, L, K, d, alpha1, alpha2, lambda, tao)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Spectral Method for Semi-supervised manifold learning
% X: samples
% Y: latent variable of labeled samples
% L: X(:,1:L) labeled samples
% K: the number of neighbors
% d: low-dimension of latent space
% 0<alpha1, alpha2<1, lambda: parameters
%
% Reference:
% Zhang, Zhenyue, Hongyuan Zha, and Min Zhang. 
% "Spectral methods for semi-supervised manifold learning." CVPR 2008.
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

Bshow=zeros(K,K,N);
for i=1:N
    Bshow(:,:,i)=BI{i};
end

BL = zeros(N);
for i=IndL
    Ii = Nei(:,i)';
    BL(Ii,Ii) = BL(Ii,Ii)+BI{i};
end

BP = zeros(N);
for i=IndP
    Ii = Nei(:,i)';
    BP(Ii,Ii) = BP(Ii,Ii)+BI{i};
end

BN = zeros(N);
for i=IndN
    Ii = Nei(:,i)';
    BN(Ii,Ii) = BN(Ii,Ii)+BI{i};
end
B = alpha1*(BL+BL')/2+(BP+BP')/2+alpha2*(BN+BN')/2;

% Create phi for labeled data set
Ye=[ones(L,1),Y'];
[Vi,Sig,Ui]=svd(Ye,0);  
% construct Gi
% compute the local orthogonal projection Bi = I-Gi*Gi' 
% that has the null space span([e,Y^T]). 
Py = eye(L)-Vi*Vi';
PyS = zeros(N);
PyS(1:L,1:L) = Py;
PyS=(PyS+PyS')/2;

beta=(lambda*N/L);
Phi = B + beta*PyS;


% % min tr(Z*Phi*Z^T)+||Z_L-Y_L||_F^2
% [U,S,V]=svd(Phi,0);
% A=U*sqrt(S);
% % min ||Z*A||_F^2+||Z_L-Y_L||_F^2
% SL=zeros(N,L);
% SL(1:L,1:L)=eye(L);
% % Z*[A,SL]=[0,Y_L]
% tmp=[A,SL];
% Z=[zeros(d,N),Y]*tmp'*pinv(tmp*tmp');

options.disp = 0; 
options.isreal = 1; 
options.issym = 1; 
[U,D] = eigs(Phi ,d+2, 0, options);  
Lambda = diag(D);
[~,J] = sort(abs(Lambda));
U = U(:,J); 
Z = U(:,2:d+1)';

Q11=U(1:L,1:d+1);
E=Y*Q11*pinv( Q11'*Q11 + tao*(norm(Q11)^2)*eye(size(Q11,2)) );
Z=E*U(:,1:d+1)';
% ZZ=[ones(1,N);Z];
% ZZL=ZZ(:,1:L);
% E=Y*ZZL'*pinv(ZZL*ZZL'+tao*norm(ZZL)*eye(size(ZZL,1)));
% Z=E*ZZ;
