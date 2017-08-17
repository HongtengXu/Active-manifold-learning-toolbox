%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Demo of active manifold learning on toy model
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear


% parameter of ML
% K-neighbors
options.K=8;
% dimension of latent space
options.d=2;
options.ML='LTSA';
options.s=4;
options.sigma = 1;

% parameter of SSML
alpha=0.03;
alpha1=2*alpha;
alpha2=alpha;
lambda=100;
tao=0.0025;
beta=0.1;


% configure
TestNum=5;
L=20:20:100;
AL = {'RD', 'GD', 'DPP', 'GC', 'HGC', 'FGC'};
SSML = {'LS','Spec'};
Noise = 2;

% evaluation
Time = zeros(length(AL),length(L), Noise, TestNum);
RelErr = zeros(length(AL),length(L), Noise, length(SSML),TestNum);

LL=zeros(size(L));

for n=1:TestNum
    for nn=1:Noise
        % data generation
        % original samples
        N = 500; % the number of samples
        t1 = random('unif',0,5*pi/3,[1,N]);
        t2 = random('unif',0,5*pi/3,[1,N]);
        % data
        X = [(3+cos(t1)).*cos(t2);...
             (3+cos(t1)).*sin(t2);...
              sin(t1)];
        X = X+((nn-1)*0.05)*rand(size(X));

        % real parameters
        Y=[t1;t2];

        T1=0:0.1*pi/3:5*pi/3;
        T2=T1;
        [x,y]=meshgrid(T1,T2);
        z1=(3+cos(x)).*cos(y);
        z2=(3+cos(x)).*sin(y);
        z3=sin(x);

        % visualization of data
        if n==1 && nn==1
            h=figure;
            hold on
            mesh(z1,z2,z3,'FaceColor','interp');
            shading faceted
            plot3(X(1,:),X(2,:),X(3,:),'m+');
            axis equal
            hold off
            savefig(h,'toymodel.fig');
            close(h)
        end

        
        for l = 1:length(L)
            Z = cell(length(SSML),length(AL));
            for i=1:length(AL)

                [Align, IndexU, IndexL, Time(i,l,nn,n)] = ...
                    ActiveManifoldLearning( X, options, AL{i}, L(l) );

                X=[X(:,IndexL),X(:,IndexU)];
                Y=[Y(:,IndexL),Y(:,IndexU)];
                YL=Y(:,1:L(l));

                for m = 1:length(SSML)
                    if m==1
                        tic;
                        Z{m,i} = LestSquareSemiSupervisedML( X, YL, L(l), ...
                            options.K, options.d, beta);
                        toc;

                    else

                        tic;
                        Z{m,i} = SpectralSemiSupervisedML( X, YL, L(l), options.K,...
                            options.d, alpha1, alpha2, lambda, tao );
                        toc;

                    end

                    RelErr(i,l,nn,m,n)=norm( Z{m,i}(:,L(l)+1:end)-Y(:,L(l)+1:end), 'fro' )...
                            /norm( Y(:,L(l)+1:end), 'fro' );
                        
                    
                    
                end

            end
            
            if RelErr(length(AL),l,nn,2,n) == min(RelErr(:,l,nn,2,n)) && LL(l)<10
                
                h=figure;
                for i=1:length(AL)
                    for m=1:length(SSML)

                        subplot(length(SSML),length(AL),(m-1)*length(AL)+i)
                        plot(t1,t2,'b.',Z{m,i}(1,L(l)+1:end),Z{m,i}(2,L(l)+1:end),'ro',...
                                Z{m,i}(1,1:L(l)),Z{m,i}(2,1:L(l)),'g*');
                        fn=sprintf('%s-%s: err=%s',AL{i}, SSML{m},...
                        num2str(RelErr(i,l,nn,m,n)));
                        title(fn)
                        axis tight
                        axis square
                        legend('real parameters', 'estimate parameters',...
                            'labeled parameters')
                    end
                end
                savefig(h,sprintf('Res_%d_%d.fig',l,LL(l)));
                close(h)
                LL(l) = LL(l)+1;
            end

        end
    end
end

save('Result2.mat','RelErr','Time');






