function [index, index2] = ActiveLearningGC( M, K )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Active learning based on Gerschgorin circle
%
% M is a N*N normal symmetric matrix
% Mout is a (N-K)*(N-K) matrix, which is obtained from deleting K columns
% and rows of Min
%
% We want to minimize the condition number of Mout, (k(Mout))
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% log(Min), so that log(k(Min))=lambda_{max}(logMin)-lambda_{min}(logMin)
logM=logm(M);

% diagonal elements
D=diag(logM);
% absolute sum of nondiagonal elements
A=sum(abs(logM),2)-D;

index=1:length(D);
index2=zeros(1,K);

upper=D+A;
lower=D-A;


for k=1:K
    ind1=find(upper==max(upper));
    ind2=find(lower==min(lower));
    
    R1=A(ind1(1));
    R2=A(ind2(1));
    
    if R1>R2
        ind=ind1(1);
    else
        ind=ind2(1);
    end
    
    index2(k) = index(ind);
    if ind==1
        index=index(ind+1:end);
        
    else if ind==length(index)
            index=index(1:end-1);
            
        else
            index=index([1:ind-1,ind+1:end]);
            
        end
    end
    tmp=M(index,index);
    try
        logM=logm(tmp);
    catch
        logM=logm(tmp+eye(size(tmp)));
    end

    % diagonal elements
    D=diag(logM);
    % absolute sum of nondiagonal elements
    A=sum(abs(logM),2)-D;
    
    upper=D+A;
    lower=D-A;
end
