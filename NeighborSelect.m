function [Nei, IndL, IndP, IndN]=NeighborSelect(X, L, K)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% X: samples
% L: X(:,1:L) are labeled samples
% K: the number of neighbors
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS 
N = size(X,2);
X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*(X'*X);
[~,index] = sort(distance);
Nei = index(1:K,:);

ind=sum( double( (Nei(:,L+1:end)-L)<=0 ) );
IndL=1:L;
IndP=find(ind>0)+L;
IndN=find(ind==0)+L;
        
