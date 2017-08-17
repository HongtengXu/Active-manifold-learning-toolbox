function [indexU, indexL] = ActiveLearningHGC( Align, L )

% diagonal elements
D=diag(Align);
% absolute sum of nondiagonal elements
A=sum(abs(Align),2)-abs(D);

lambda = max([0.1,-min(D-A)]);

Psi = Align+lambda*eye(size(Align)); 


C = diag(Psi);
R = sum(abs(Psi),2)-abs(C);
S = R;
indexU = 1:size(Psi,1);
indexL = [];
[UpperP, ID] = sort(C+R,'descend');
%indexU = indexU(ID);

for l=1:L
    
    l1 = ID(1);
    [~,ind1] = min( C(indexU)-S(indexU) );
    IND = [l1, indexU(ind1)];
%     if l==1
%         
%     else
%         [~,ind2] = max( R(indexU)-S(indexU) );
%         IND = [l1, indexU(ind1), indexU(ind2)];
%     end
    
    
    obj = zeros(length(IND),1);
    for i=1:length(IND)
        tmp=1:size(Psi,1);
        tmp([indexL,IND(i)])=[];
        
        Si = S(tmp)-abs(Psi(tmp,IND(i)));
        v1 = min( C(tmp)-Si );
        v2 = max( R(tmp)-Si );
        if i==1
            v3 = UpperP(2);
        else
            v3 = UpperP(1);
        end
        obj(i) = v3/(v1*v2);
    end
    
    [~, index] = min(obj);
    
    if index(1)==1
        ID(1)=[];
        UpperP(1)=[];
    end
    
    indexL = [indexL, IND(index(1))];
    indexU=1:size(Psi,1);
    indexU(indexL)=[];
    S(indexU) = S(indexU) - abs(Psi(indexU, IND(index(1))));
    
end