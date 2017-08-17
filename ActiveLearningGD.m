function [indexU, indexL] = ActiveLearningGD( X, options, L )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Active learning via maximzing the minimum geodesic distance between
% landmark points
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, N] = size(X);

P = randperm(N);
indexU = 1:N;
indexL = zeros(1, L);
indexL(1:options.s) = P(1:options.s);

m = zeros(N,1);
for j=1:N
    res = sum( (repmat( X(:,j), [1,options.s] ) - X(:, indexL(1:options.s))).^2 );
    m(j) = min(res);
end

for i=options.s+1 : L
    [~,ind] = max(m);
    indexL(i) = ind(1);
    
    for j=1:N
        res = sum( (X(:,j) - X(:,indexL(i))).^2 );
        m(j) = min([m(j), res]);
    end
end

indexU(indexL)=0;
indexU = find(indexU~=0);




