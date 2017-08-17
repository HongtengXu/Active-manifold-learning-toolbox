function [indexU, indexL] = ActiveLearningFGC( Align, L )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Active learning via self-paced condition number minimization
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% diagonal elements
D=diag(Align);
% absolute sum of nondiagonal elements
A=sum(abs(Align),2)-abs(D);

lambda = -min(D-A);

Phi = Align+lambda*eye(size(Align)); 


indexU=1:length(D);
indexL=zeros(1,L);

upper=lambda+D+A;
lower=lambda+D-A;

Tmp = Phi-diag(diag(Phi));
DD = diag(Phi);

for k=1:L
    bound=zeros(4,1);
    IND=bound;
    

    [~, indupper]=max(upper);
    [~, indlower]=min(lower);
    IND(1)=indupper(1);
    IND(2)=indlower(1);
    
    [~, ind1]=max(abs(Tmp(indupper(1),:)));
    [~, ind2]=max(abs(Tmp(indlower(1),:)));
    IND(3)=ind1(1);
    IND(4)=ind2(1);
    
    for i=1:4
        
        ind = true(size(Tmp(1,:)));
        ind(IND(i)) = false;
        tmp = Tmp(ind, ind);
        % diagonal
        D = DD(ind);
        
        % absolute sum of nondiagonal elements
        A=sum(abs(tmp),2);
        
        upper=D+A;
        lower=D-A;
        
        bound(i) = max(upper)/min(lower);
    end
    
    
%     IND1= true(size(upper));
%     IND1(indupper(1))=false;
%     IND2= true(size(lower));
%     IND2(indlower(1))=false;
%     
%     uppernew1 = upper - abs(Tmp(:,indupper));
%     lowernew1 = lower + abs(Tmp(:,indupper));
%     bound1 = uppernew1./lowernew1;
%     bound(1) = max(uppernew1(IND1))/min(lowernew1(IND1));%max(bound1(IND1));
%     IND(1) = indupper(1);
%     
%     uppernew2 = upper - abs(Tmp(:,indlower));
%     lowernew2 = lower + abs(Tmp(:,indlower));
%     bound2 = uppernew2./lowernew2;
%     bound(2) = max(uppernew2(IND2))/min(lowernew2(IND2));%max(bound2(IND2));
%     IND(2) = indlower(1);
%     
%     [value1U, ind1]=max(abs(Tmp(indupper(1),:)));
%     value1L = abs(Tmp(indlower(1),ind1(1)));
%     bound(3) = ( valueU(1)-value1U(1) )/(valueL(1)+value1L(1));
%     IND(3) = ind1(1);
%     
%     [value2L, ind2]=max(abs(Tmp(indlower(1),:)));
%     value2U = abs(Tmp(indupper(1),ind2(1)));
%     bound(4) = ( valueU(1)-value2U(1) )/(valueL(1)+value2L(1));
%     IND(4) = ind2(1);
    
    [~,ID]=min(bound);
    ind = IND(ID);

    
    
    indexL(k) = indexU(ind);
    if ind==1
        indexU=indexU(ind+1:end);
        
    else if ind==length(indexU)
            indexU=indexU(1:end-1);
            
        else
            indexU=indexU([1:ind-1,ind+1:end]);
            
        end
    end
    Tmp=Phi(indexU,indexU);
    DD=diag(Tmp);
    Tmp=Tmp-diag(diag(Tmp));
    
    
%     % diagonal elements
%     D=diag(Tmp);
%     % absolute sum of nondiagonal elements
%     A=sum(abs(Tmp),2)-abs(D);
%     
%     upper=D+A;
%     lower=D-A;
end