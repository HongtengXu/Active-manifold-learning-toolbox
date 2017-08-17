function [indexU, indexL] = ActiveLearningDPP( X, options, L )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Active Learning based on Determinantal Point Processes
%
% Reference:
% Wachinger, Christian, and Polina Golland. 
% "Diverse Landmark Sampling from Determinantal Point Processes for Scalable Manifold Learning." 
% arXiv preprint arXiv:1503.03506 (2015).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,N]=size(X);
indexU = 1:N;
indexL = zeros(1,L);
Dis = ones(1,N);


for k=1:L
    
    pd = makedist('Multinomial', 'probabilities', Dis./sum(Dis));
    
    % select sample
    i = random(pd);
    indexL(k) = i;
    
    
    % update distribution
    Dis(i) = 0;
    res = sum( ( repmat(X(:,i),[1,N])-X ).^2 );
    [~,ind] = sort(res, 'ascend');
    ind = ind(1:options.K);
    Dis(ind) = Dis(ind).*updatefunction(res(ind), options.sigma);
      
end

indexU(indexL)=0;
indexU = find(indexU~=0);

end


function f = updatefunction(res, sigma)
    f = 1 - exp( -res./(2*sigma^2) );
end
