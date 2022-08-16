%%
clear all 
close all
clc
addpath('fsparse')
rng('shuffle')


%% PARAMETRIZATION
rmin=1.1;

%Parameters
E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3 
mload = 5e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; %Load
nu = 0.33; % -
penal = 3; % Penalty


%Number of elements
lx=300;
ly=100;
nelx = 30; 
nely = 10; 
a = 0.49*lx/nelx; %mm half of the element length in x 
b = 0.49*ly/nely; %mm half of the element length in y
h = 1; %mm thickness of the plate 

% Node coordinates 
% If even elements in x and y direction
coordx = -2*a*nelx/2:2*a:2*a*nelx/2; 
coordy = -2*b*nely/2:2*b:2*b*nely/2;
%If uneven
if mod(nelx,2) ~= 0 && mod(nely,2) ~= 0
    coordx = -2*a*(nelx-1)/2-a:2*a:2*a*(nelx-1)/2+a;  
    coordy = -2*b*(nely-1)/2-b:2*b:2*b*(nely-1)/2+b;
elseif mod(nelx,2) ~= 0 
    coordx = -2*a*(nelx-1)/2-a:2*a:2*a*(nelx-1)/2+a;       
elseif mod(nely,2) ~= 0
    coordy = -2*b*(nely-1)/2-b:2*b:2*b*(nely-1)/2+b;
end 
[coordX,coordY] = meshgrid(coordx,coordy);


%2d Stiffness Matrix 
gamma = a/b;  
k = [(1+nu)*gamma,(1-3*nu)*gamma,2+(1-nu)*gamma^2,2*gamma^2+(1-nu),(1-nu)*gamma^2-4,(1-nu)*gamma^2-1,4*gamma^2 - (1-nu), gamma^2 - (1-nu)];
KE = E*h/((1-nu^2)*24*gamma)* ...
    [ 4*k(3),  3*k(1),  2*k(5), -3*k(2), -2*k(3), -3*k(1), -4*k(6),  3*k(2);
      3*k(1),  4*k(4),  3*k(2),  4*k(8), -3*k(1), -2*k(4), -3*k(2), -2*k(7);
      2*k(5),  3*k(2),  4*k(3), -3*k(1), -4*k(6), -3*k(2), -2*k(3),  3*k(1);
     -3*k(2),  4*k(8), -3*k(1),  4*k(4),  3*k(2), -2*k(7),  3*k(1), -2*k(4);  
     -2*k(3), -3*k(1), -4*k(6),  3*k(2),  4*k(3),  3*k(1),  2*k(5), -3*k(2);
     -3*k(1), -2*k(4), -3*k(2), -2*k(7),  3*k(1),  4*k(4),  3*k(2),  4*k(8); 
     -4*k(6), -3*k(2), -2*k(3)   3*k(1),  2*k(5),  3*k(2),  4*k(3), -3*k(1);
      3*k(2), -2*k(7),  3*k(1), -2*k(4), -3*k(2),  4*k(8), -3*k(1),  4*k(4)]; 

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([-2 -1 2*nely+[0 1 2 3] 0 1 ],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);


%% PREPARE FILTER
iH = ones((nelx)*nely*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
  for j1 = 1:nely
    e1 = (i1-1)*nely+j1;
    for i2 = max(i1-(ceil(rmin)-1),2):min(i1+(ceil(rmin)-1),nelx-1)
      for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
        e2 = (i2-1)*nely+j2;
        k = k+1;
        iH(k) = e1;
        jH(k) = e2;
        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
      end
    end
  end
end
H = fsparse(iH,jH,sH);
Hs = sum(H,2);
H = H(nely+1:nely*(nelx-1),nely+1:nely*(nelx-1));
Hs = Hs(nely+1:nely*(nelx-1));


%% Guyan
K = fsparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1),0);
K_g = fsparse(2*2*(nely+1)+6,2*2*(nely+1)+6,0);
[m,n]= size(K);
K_ = fsparse(m+6,n+6,0);
K_(4:end-3,4:end-3) = K;
alldofs0_g   = [1:length(K_)];
mdofs_g = [1:(2*(nely+1))+3,length(K_)-(2*(nely+1))+1-3:length(K_)];
sdofs_g = setdiff(alldofs0_g,mdofs_g);
alldofs_g = [mdofs_g, sdofs_g];   



%% RBE2
alldofs0_r = 1:length(K_g);                   %All dofs in original order
sdofs_r = [4:length(K_g)-3];             %Dofs that are to be removed
mdofs_r = setdiff(alldofs0_r,sdofs_r);            %Dofs that remain
alldofs_r = [mdofs_r,sdofs_r];                    %New order, sdofs are at the end
newdofs_r = zeros(length(alldofs_r),1);           %Nonzeros will remove the condensed nodes
newdofs_r(mdofs_r) =  1:length(mdofs_r);          %Accesing the new order with the old one
newdofs_r(4:end-3) = 7:length(newdofs_r(6:end-3))+8;


%Coordinates of the free nodes 
coordRBE = [-lx/2,lx/2;
            0,0;
            0,0];

C = fsparse(length(sdofs_r),length(K_g),0);    
% Left Side
for n = 1:nely+1
    C(2*(n-1)+1,1) =1;                                                      % First DOF of independent node
    C(2*(n-1)+2,2) =1;                                                      % Second DOF of independent node
    
    C_t = cross([0;0;1],[coordX(n,1) - coordRBE(1,1); coordY(n,1);0]);      % Third DOF of independent node
    C(2*(n-1)+1,3) =C_t(1);                                                 % Third DOF of independent node
    C(2*(n-1)+2,3) = C_t(2);                                                % Third DOF of independent node
    
    C(2*(n-1)+1,3+(n-1)*2+1) =-1;                                           % Dependent node of 2d elements to be removed
    C(2*(n-1)+2,3+n*2) = -1;                                                % Dependent node of 2d elements to be removed
end 
% Right Side
for n = 1:nely+1
    C(2*(n-1)+1+(nely+1)*2,4*(nely+1)+4) =1;                                % First DOF of independent node
    C(2*(n-1)+2+(nely+1)*2,4*(nely+1)+5) =1;                                % Second DOF of independent node
    
    C_t = cross([0;0;1],[coordX(n,end) - coordRBE(1,2); coordY(n,end);0]);  % Third DOF of independent node
    C(2*(n-1)+1+(nely+1)*2,4*(nely+1)+6) =C_t(1);                           % Third DOF of independent node
    C(2*(n-1)+2+(nely+1)*2,4*(nely+1)+6) = C_t(2);                          % Third DOF of independent node
    
    C(2*(n-1)+1+(nely+1)*2,(nely+1)*2+3 + 2*(n-1)+1) =-1;                   % Dependent node of 2d elements to be removed
    C(2*(n-1)+2+(nely+1)*2,(nely+1)*2+3 + 2*(n-1)+2) = -1;                  % Dependent node of 2d elements to be removed
end 

Q = fsparse(size(C,1),1,0);                       
%Set up model for the unconstrained case
Tsm = -C(:,sdofs_r)\C(:,mdofs_r);
Ti = speye(length(mdofs_r));
T_r = [Ti;Tsm]; 
T_rt = transpose(T_r);


x = ones(nely,nelx);
sK = reshape(KE(:)*(x(:)'.^penal),64*nelx*nely,1);
K = fsparse(iK,jK,sK); K = (K+K')/2;
%Guyan 
[t,r]= size(K);
K_ = fsparse(t+6,r+6,0);
K_(4:end-3,4:end-3) = K;
Kss = K_(sdofs_g,sdofs_g);
Ksm = K_(sdofs_g,mdofs_g);
T_g = [speye(length(mdofs_g)); -Kss\Ksm];
T_gt = transpose(T_g);
T_rg = T_g*T_r(newdofs_r,1:end);
T_rgt = transpose(T_rg); 
K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;

%Upper and Lower Stiffness Bounds 
x_lb = -0.1*[K_rg(1,1);K_rg(2,2);K_rg(3,3);K_rg(6,6)]; x_lb = full(x_lb);
x_ub = 1.1*[K_rg(1,1);K_rg(2,2);K_rg(3,3);K_rg(6,6)]; x_ub = full(x_ub);


%% Sampling
N_class=[500, 1000:500:5500];
frac = length(N_class)-1;

X = NaN(4,N_class(end));
y = NaN(1,N_class(end));
y_class = -1*ones(1,N_class(end));
X_class = zeros(4,N_class(end));


DoE = lhsdesign(N_class(1),2)'; 
deltax = 0.2*(1 - 1e-3); %0.75
density = 1e-3 + (1 - 1e-3 - deltax)*linspace(0,1,N_class(1))'.*ones(N_class(1),nely/2*(nelx-2))+ lhsdesign(N_class(1),nely/2*(nelx-2))*deltax;  


% Physical Seed 
parfor i = 1:length(DoE)
        xtemp = density(i,:);
        x = ones(nely,nelx);
        x(1:nely/2,2:nelx-1) =   reshape(xtemp,nely/2,nelx-2);
        x(nely/2+1:end,2:nelx-1) = x(linspace(nely/2,1,nely/2),2:nelx-1); 
        
        %INITIAL ANALYSIS
        sK = reshape(KE(:)*(x(:)'.^penal),64*nelx*nely,1);
        K = fsparse(iK,jK,sK); K = (K+K')/2;
        m = sum(x(:))/(nely*nelx)*100;

        %Guyan 
        [t,r]= size(K);
        K_ = fsparse(t+6,r+6,0);
        K_(4:end-3,4:end-3) = K;
        Kss = K_(sdofs_g,sdofs_g);
        Ksm = K_(sdofs_g,mdofs_g);
        T_g = [speye(length(mdofs_g)); -Kss\Ksm];
        T_rg = T_g*T_r(newdofs_r,1:end);
        T_rgt = transpose(T_rg); 
        K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;
        X_class(:,i) = [K_rg(1,1);K_rg(2,2);K_rg(3,3);K_rg(6,6)]; 
        y_class(:,i) = 1;
end 

plotActiveLearning(X_class,y_class,100)  

SVM = fitcsvm(X_class(:,1:N_class(1))',y_class(1:N_class(1)),'KernelFunction','gaussian','KernelScale','auto',...
    'Standardize',true,'OutlierFraction',0.15);

%Sampling
N_temp_0 = 1e5;
X_temp_1 = x_lb(1) + rand(N_temp_0,1)*(x_ub(1)-x_lb(1));
X_temp_2 = x_lb(2) + rand(N_temp_0,1)*(x_ub(2)-x_lb(2));
X_temp_3 = x_lb(3) + rand(N_temp_0,1)*(x_ub(3)-x_lb(3));
X_temp_4 = x_lb(4) + rand(N_temp_0,1)*(x_ub(4)-x_lb(4));

[~,score] = predict(SVM,[X_temp_1,X_temp_2,X_temp_3,X_temp_4]);
[sortedVals,indexes] = sort(score,'descend');
X_class(:,N_class(1)+1:N_class(2)) =  [X_temp_1(indexes(1:N_class(2)-N_class(1))),X_temp_2(indexes(1:N_class(2)-N_class(1))),...
    X_temp_3(indexes(1:N_class(2)-N_class(1))),X_temp_4(indexes(1:N_class(2)-N_class(1)))]';


plotActiveLearning(X_class,y_class,101)  

tic
for i=1:frac
    parfor j=N_class(i)+1:N_class(i+1)
       
         K_0= X_class(:,j)'; 

         % INITIALIZE OPTIMIZATION
        epsilon=1e-2;
        feasibleflag = 0;
        classifierflag = 0;
        exitflag = NaN;
        MaxIterations= 1300;
        ConstraintTolerance = epsilon*1e-1;
        StepTolerance = 1e-3;
        x = ones(nely,nelx);
        x(:,2:nelx-1) = 0.5*ones(nely,(nelx-2));  
        loop = 0; 
        xold1 =  reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);      % For the MMA-Algorithm
        xold2 =  reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]); 
        mm = 8;                                         % Number of constraints
        nn=nelx*nely - 2*nely;                           % Number of designvariables
        aa0=1;                   
        aa=zeros(mm,1);
        cc=1e3*ones(mm,1);
        dd=zeros(mm,1);
        xmin = ones(nn,1)*0.001;         % Lower bounds of design variables
        low = xmin;
        xmax = ones(nn,1);               % Upper bounds of design variables
        upp = xmax;



        %INITIAL ANALYSIS
        sK = reshape(KE(:)*(x(:)'.^penal),64*nelx*nely,1);
        K = fsparse(iK,jK,sK); K = (K+K')/2;
        m = sum(x(:))/(nely*nelx)*100;

        %Guyan 
        [t,r]= size(K);
        K_ = fsparse(t+6,r+6,0);
        K_(4:end-3,4:end-3) = K;
        Kss = K_(sdofs_g,sdofs_g);
        Ksm = K_(sdofs_g,mdofs_g);
        T_g = [speye(length(mdofs_g)); -Kss\Ksm];
        T_rg = T_g*T_r(newdofs_r,1:end);
        T_rgt = transpose(T_rg); 
        K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;

        % GRADIENTS
        dK_rg = zeros(nely*(nelx-2),mm/2); el = 0;
        for elx = 2:nelx-1
          for ely = 1:nely/2
            el = el+1;
            dK_ = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,0);
            n1 = (nely+1)*(elx-1)+ely; 
            n2 = (nely+1)* elx   +ely;
            edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
            dK_(edof+3,edof+3) =  penal*x(ely,elx)^(penal-1)*KE;
            dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg;   
            dK_rg(el,:) = dK_rg_t([1,8,15,36])./K_0;
          end
            dK_rg(el +1: el + nely/2,:) = dK_rg(linspace(el,el-nely/2+1,(nely/2)),:);
            el = el + nely/2;
        end 


        for ii=1:mm/2
            dK_rg(:,ii) = H*(reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]).*dK_rg(:,ii))./Hs./reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);
        end 

        dm = ones(nely,nelx-2)/(nely*nelx)*100;
        dm= reshape(dm,[(nelx-2)*nely,1]);  


        %Both sides
        k = zeros(mm,1);
        k(1:mm/2) = (K_rg([1,8,15,36]) - K_0)./K_0 - epsilon;
        k(mm/2+1:end) = (K_0 - K_rg([1,8,15,36]))./K_0- epsilon;

        dk = zeros((nelx-2)*nely,mm);
        dk(:,1:mm/2) = dK_rg;
        dk(:,mm/2+1:end) = -dK_rg;

        % MMA OPTIMIZATION
        xval =  reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);
        f0val =m;     
        df0dx= dm; 
        df0dx2= 0*df0dx;
        fval=k;          
        dfdx=dk';
        dfdx2=0*dfdx;  

        % START ITERATION

        conv = 0;
        while conv == 0
            loop = loop + 1;

            % MMA OPTIMIZATION
            [xmma,ymma,zmma,lam,xsi,eta,mu,zet,ss,low,upp] = ...
                mmasub_old(mm,nn,loop,xval,xmin,xmax,xold1,xold2, ...
                f0val,df0dx,df0dx2,fval,dfdx,dfdx2,low,upp,aa0,aa,cc,dd);

            f0valold = f0val;    
            xold2 = xold1;
            xold1 = xval;
            xval = xmma;
            x(:,2:nelx-1) = reshape(xval,[nely,nelx-2]);


            %INITIAL ANALYSIS
            sK = reshape(KE(:)*(x(:)'.^penal),64*nelx*nely,1);
            K = fsparse(iK,jK,sK); K = (K+K')/2;
            m = sum(x(:))/(nely*nelx)*100;

            %Guyan 
            [t,r]= size(K);
            K_ = fsparse(t+6,r+6,0);
            K_(4:end-3,4:end-3) = K;
            Kss = K_(sdofs_g,sdofs_g);
            Ksm = K_(sdofs_g,mdofs_g);
            T_g = [speye(length(mdofs_g)); -Kss\Ksm];
            T_rg = T_g*T_r(newdofs_r,1:end);
            T_rgt = transpose(T_rg); 
            K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;
            K_rg = K_rg([1,8,15,36]);

            % GRADIENTS
            dK_rg = zeros(nely*(nelx-2),mm/2); el = 0;
            for elx = 2:nelx-1
              for ely = 1:nely/2
                el = el+1;
                dK_ = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,0);
                n1 = (nely+1)*(elx-1)+ely; 
                n2 = (nely+1)* elx   +ely;
                edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
                dK_(edof+3,edof+3) =  penal*x(ely,elx)^(penal-1)*KE;
                dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg; 
                dK_rg(el,:) = dK_rg_t([1,8,15,36])./K_0;
              end
                dK_rg(el +1: el + nely/2,:) = dK_rg(linspace(el,el-nely/2+1,(nely/2)),:);
                el = el + nely/2;
            end 


            for ii=1:mm/2
                dK_rg(:,ii) = H*(reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]).*dK_rg(:,ii))./Hs./reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);
            end 

            dm = ones(nely,nelx-2)/(nely*nelx)*100;
            dm= reshape(dm,[(nelx-2)*nely,1]);  


            %Both sides
            k = zeros(mm,1);
            k(1:mm/2) = (K_rg - K_0)./K_0 - epsilon;
            k(mm/2+1:end) = (K_0 - K_rg)./K_0- epsilon;

            dk = zeros((nelx-2)*nely,mm);
            dk(:,1:mm/2) = dK_rg;
            dk(:,mm/2+1:end) = -dK_rg;

            % MMA OPTIMIZATION
            xval =  reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);
            f0val =m;     
            df0dx= dm; 
            df0dx2= 0*df0dx;
            fval=k;          
            dfdx=dk';
            dfdx2=0*dfdx;  


            % Convergence Check
            change_x = max(abs(xval-xold1));
            feasible_f = max(k);


            if (feasible_f < ConstraintTolerance) && (y_class(j) ~= 1)
                y_class(j) = 1;
                X_class(:,j) = K_rg;  
            end

            %If the final design is not feasible and not mass optimal:
            %Classifer Sample
            if loop >= MaxIterations && feasible_f > ConstraintTolerance            
               conv =1;
               exitflag = -2;

            %If the final design is feasible, but not mass optimal: Classifier
            %Sample
            elseif loop >= MaxIterations && feasible_f < ConstraintTolerance
               conv =1;
               exitflag = 0;
            %If a design was never feasible in the optimization, the algorithm
            %is aborted: Classifier Sample
            elseif (change_x < StepTolerance) && feasible_f > ConstraintTolerance  && y_class(j) ~= 1
                conv=1;
                exitflag =-2; 

            % If the steptolerance is below the limit and the design is
            % feasible: Mass Sample
            elseif (change_x < StepTolerance) && feasible_f < ConstraintTolerance 
                conv=1;
                exitflag =2;
                y_class(j) = 1;
                X_class(:,j) = K_rg;  
                X(:,j)  = K_rg;
                y(j) = m;
            end
        end 


    end 
    
    if i== frac
        break
    end
    
    
    C = ones(2,2)-eye(2,2);
    C(1,2) =2;
    SVM = fitcsvm(X_class(:,1:N_class(i+1))',y_class(1:N_class(i+1)),'Standardize',true,'KernelFunction','gaussian','Cost',C,'OptimizeHyperparameters','auto',...
       'HyperparameterOptimizationOptions',struct('UseParallel',true,'MaxObjectiveEvaluations',100)); 
    gcf; close;
    gcf; close;  
    
    
    plotActiveLearning(X_class,y_class,i)  

    N_temp = round(i^4*(1e6-N_temp_0 )/(frac-1)^4 +  N_temp_0); %1e7 || ^2
    X_temp = zeros(4,N_temp);
    X_temp(1,:) = x_lb(1) + rand(1,N_temp)*(x_ub(1)-x_lb(1));
    X_temp(2,:) = x_lb(2) + rand(1,N_temp)*(x_ub(2)-x_lb(2));
    X_temp(3,:) = x_lb(3) + rand(1,N_temp)*(x_ub(3)-x_lb(3));
    X_temp(4,:) = x_lb(4) + rand(1,N_temp)*(x_ub(4)-x_lb(4));
    [label,distance] = predict(SVM,X_temp');
    [sortedVals,indexes] = sort(abs(distance(:,1)));
    X_class(:,N_class(i+1)+1:N_class(i+2)) = X_temp(:,indexes(1:N_class(i+2)-N_class(i+1)));
end 
    

plotActiveLearning(X_class,y_class,frac)  


C = ones(2,2)-eye(2,2);
C(1,2) =2;
SVM = fitcsvm(X_class',y_class,'Standardize',true,'KernelFunction','gaussian','Cost',C,'OptimizeHyperparameters','auto',...
       'HyperparameterOptimizationOptions',struct('UseParallel',true,'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100)); 
gcf; close;
gcf; close;  


X_temp = X(:,~isnan(y));
y_temp = y(~isnan(y)); 

N_mass = length(y_temp);
    
    
%% Sampling Mass 
N= 6000;
X = NaN(4,N);
X(:,1:N_mass)= X_temp;
y = NaN(1,N);
y(:,1:N_mass)= y_temp;
X_class_2 = zeros(4,N);
y_class_2 = -1*ones(1,N);
t_list = zeros(N,1);

parfor j=N_mass+1:N
    
    %Only choose feasible designs
    label = -1;
    tic
    while label == -1
        X(:,j) = x_lb + rand(4,1).*(x_ub-x_lb);
        label = predict(SVM,X(:,j)');
    end
    t_list(j,1) = toc;

    K_0= X(:,j)'; 

     % INITIALIZE OPTIMIZATION
    epsilon=1e-2;
    feasibleflag = 0;
    classifierflag = 0;
    exitflag = NaN;
    MaxIterations= 1300;
    ConstraintTolerance = epsilon*1e-1;
    StepTolerance = 1e-3;
    x = ones(nely,nelx);
    x(:,2:nelx-1) = 0.5*ones(nely,(nelx-2));  
    loop = 0; 
    xold1 =  reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);      % For the MMA-Algorithm
    xold2 =  reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]); 
    mm = 8;                                         % Number of constraints
    nn=nelx*nely - 2*nely;                           % Number of designvariables
    aa0=1;                   
    aa=zeros(mm,1);
    cc=1e3*ones(mm,1);
    dd=zeros(mm,1);
    xmin = ones(nn,1)*0.001;         % Lower bounds of design variables
    low = xmin;
    xmax = ones(nn,1);               % Upper bounds of design variables
    upp = xmax;



    %INITIAL ANALYSIS
    sK = reshape(KE(:)*(x(:)'.^penal),64*nelx*nely,1);
    K = fsparse(iK,jK,sK); K = (K+K')/2;
    m = sum(x(:))/(nely*nelx)*100;

    %Guyan 
    [t,r]= size(K);
    K_ = fsparse(t+6,r+6,0);
    K_(4:end-3,4:end-3) = K;
    Kss = K_(sdofs_g,sdofs_g);
    Ksm = K_(sdofs_g,mdofs_g);
    T_g = [speye(length(mdofs_g)); -Kss\Ksm];
    T_rg = T_g*T_r(newdofs_r,1:end);
    T_rgt = transpose(T_rg); 
    K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;

    % GRADIENTS
    dK_rg = zeros(nely*(nelx-2),mm/2); el = 0;
    for elx = 2:nelx-1
      for ely = 1:nely/2
        el = el+1;
        dK_ = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,0);
        n1 = (nely+1)*(elx-1)+ely; 
        n2 = (nely+1)* elx   +ely;
        edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
        dK_(edof+3,edof+3) =  penal*x(ely,elx)^(penal-1)*KE;
        dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg;   
        dK_rg(el,:) =dK_rg_t([1,8,15,36])./K_0;
      end
        dK_rg(el +1: el + nely/2,:) = dK_rg(linspace(el,el-nely/2+1,(nely/2)),:);
        el = el + nely/2;
    end 


    for i=1:mm/2
        dK_rg(:,i) = H*(reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]).*dK_rg(:,i))./Hs./reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);
    end 

    dm = ones(nely,nelx-2)/(nely*nelx)*100;
    dm= reshape(dm,[(nelx-2)*nely,1]);  


    %Both sides
    k = zeros(mm,1);
    k(1:mm/2) = (K_rg([1,8,15,36]) - K_0)./K_0 - epsilon;
    k(mm/2+1:end) = (K_0 - K_rg([1,8,15,36]))./K_0- epsilon;

    dk = zeros((nelx-2)*nely,mm);
    dk(:,1:mm/2) = dK_rg;
    dk(:,mm/2+1:end) = -dK_rg;

    % MMA OPTIMIZATION
    xval =  reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);
    f0val =m;     
    df0dx= dm; 
    df0dx2= 0*df0dx;
    fval=k;          
    dfdx=dk';
    dfdx2=0*dfdx;  

    % START ITERATION

    conv = 0;
    while conv == 0
        loop = loop + 1;

        % MMA OPTIMIZATION
        [xmma,ymma,zmma,lam,xsi,eta,mu,zet,ss,low,upp] = ...
            mmasub_old(mm,nn,loop,xval,xmin,xmax,xold1,xold2, ...
            f0val,df0dx,df0dx2,fval,dfdx,dfdx2,low,upp,aa0,aa,cc,dd);

        f0valold = f0val;    
        xold2 = xold1;
        xold1 = xval;
        xval = xmma;
        x(:,2:nelx-1) = reshape(xval,[nely,nelx-2]);


        %INITIAL ANALYSIS
        sK = reshape(KE(:)*(x(:)'.^penal),64*nelx*nely,1);
        K = fsparse(iK,jK,sK); K = (K+K')/2;
        m = sum(x(:))/(nely*nelx)*100;

        %Guyan 
        [t,r]= size(K);
        K_ = fsparse(t+6,r+6,0);
        K_(4:end-3,4:end-3) = K;
        Kss = K_(sdofs_g,sdofs_g);
        Ksm = K_(sdofs_g,mdofs_g);
        T_g = [speye(length(mdofs_g)); -Kss\Ksm];
        T_rg = T_g*T_r(newdofs_r,1:end);
        T_rgt = transpose(T_rg); 
        K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;
        K_rg = K_rg([1,8,15,36]);

        % GRADIENTS
        dK_rg = zeros(nely*(nelx-2),mm/2); el = 0;
        for elx = 2:nelx-1
          for ely = 1:nely/2
            el = el+1;
            dK_ = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,0);
            n1 = (nely+1)*(elx-1)+ely; 
            n2 = (nely+1)* elx   +ely;
            edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
            dK_(edof+3,edof+3) =  penal*x(ely,elx)^(penal-1)*KE;
            dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg; 
            dK_rg(el,:) = dK_rg_t([1,8,15,36])./K_0;
          end
            dK_rg(el +1: el + nely/2,:) = dK_rg(linspace(el,el-nely/2+1,(nely/2)),:);
            el = el + nely/2;
        end 


        for i=1:mm/2
            dK_rg(:,i) = H*(reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]).*dK_rg(:,i))./Hs./reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);
        end 

        dm = ones(nely,nelx-2)/(nely*nelx)*100;
        dm= reshape(dm,[(nelx-2)*nely,1]);  


        %Both sides
        k = zeros(mm,1);
        k(1:mm/2) = (K_rg - K_0)./K_0 - epsilon;
        k(mm/2+1:end) = (K_0 - K_rg)./K_0- epsilon;

        dk = zeros((nelx-2)*nely,mm);
        dk(:,1:mm/2) = dK_rg;
        dk(:,mm/2+1:end) = -dK_rg;

        % MMA OPTIMIZATION
        xval =  reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);
        f0val =m;     
        df0dx= dm; 
        df0dx2= 0*df0dx;
        fval=k;          
        dfdx=dk';
        dfdx2=0*dfdx;  



        % Convergence Check
        change_x = max(abs(xval-xold1));
        feasible_f = max(k);

        if (feasible_f < ConstraintTolerance)
            y_class_2(j) = 1;
            X_class_2(:,j) = K_rg;  
        end

        %If the final design is not feasible and not mass optimal:
        %Classifer Sample
        if loop >= MaxIterations && feasible_f > ConstraintTolerance            
           conv =1;
           exitflag = -2;

        %If the final design is feasible, but not mass optimal: Classifier
        %Sample
        elseif loop >= MaxIterations && feasible_f < ConstraintTolerance
           conv =1;
           exitflag = 0;
        %If a design was never feasible in the optimization, the algorithm
        %is aborted: Classifier Sample
        elseif (change_x < StepTolerance) && feasible_f > ConstraintTolerance  && y_class_2(j) ~= 1
            conv=1;
            exitflag =-2; 

        % If the steptolerance is below the limit and the design is
        % feasible: Mass Sample
        elseif (change_x < StepTolerance) && feasible_f < ConstraintTolerance 
            conv=1;
            exitflag =2;
            y_class_2(j) = 1;
            X_class_2(:,j) =  K_rg;  
            X(:,j)  =  K_rg;  
            y(j) = m;
        end
    end

end 
toc     
    

plotActiveLearning(X,y,1000)  


% Mass Samples
X = X(:,~isnan(y));
y = y(~isnan(y));
%Add Negative Support Vector as Mass Samples, m=120
X_2 = SVM.X(SVM.IsSupportVector,:)';
label = predict(SVM,X_2');
X_2 = X_2(:,label==-1);
ratio = 0.02*length(X)/length(X_2); % 0.05-->0.02
[idx,~,~] = dividerand(length(X_2),ratio,1-ratio,0);
X_2 = X_2(:,idx);
[idx2,distance2]= knnsearch(X',X_2','Distance','seuclidean');
y_2 = 1.1*y(idx2); y_2(distance2 ==0) = y(distance2 ==0); %1.2
X_3 =  X_class_2(:,y_class_2(N_mass+1:end) == -1);
[idx3,distance3]= knnsearch(X',X_3','Distance','seuclidean');
y_3 = 1.2*y(idx3);
X_4 = [X,X_2,X_3];
y_4 = [y,y_2,y_3];
 
samples = [X_4;y_4];
save("Ergebnis1117_14\Samples_Mass\samples_" + num2str(length(samples)) + "_lx_" + num2str(lx) + "_ly_" + num2str(ly)+ "_nelx_" + num2str(nelx) + "_nely_" + num2str(nely) + '_border','samples');

plotActiveLearning(X_3,y_3,1001) 

%Classifier Samples 
% 1) Add false positive estimates from classifier
y_class_2 = y_class_2(N_mass+1:end);
y_class_3 = y_class_2(y_class_2 == -1);
X_class_2 = X_class_2(:,N_mass+1:end);
X_class_3 = X_class_2(:,y_class_2 == -1);

y_class_4 = [y_class, y_class_3];
X_class_4 = [X_class,X_class_3];

samples_class = [X_class_4;y_class_4];
save("Ergebnis1117_14\Samples_Classification\samples_" + num2str(length(y_class_4)) + "_lx_" + num2str(lx) + "_ly_"  ...
    + num2str(ly)+ "_nelx_" + num2str(nelx) + "_nely_" + num2str(nely),'samples_class');
plotActiveLearning(X_class_4,y_class_4,200)  



function plotActiveLearning(X_class,y_class,i)  
    figure(i)
    subplot(3,3,1)
    scatter(X_class(1,:),X_class(2,:),20,y_class,'filled');xlabel('k11');ylabel('k22');
    subplot(3,3,2)
    scatter(X_class(1,:),X_class(3,:),20,y_class,'filled');xlabel('k11');ylabel('k33');
    subplot(3,3,3)
    scatter(X_class(1,:),X_class(4,:),20,y_class,'filled');xlabel('k11');ylabel('k66');


    subplot(3,3,4)
    scatter(X_class(2,:),X_class(3,:),20,y_class,'filled');xlabel('k22');ylabel('k33');
    subplot(3,3,5)
    scatter(X_class(2,:),X_class(4,:),20,y_class,'filled');xlabel('k22');ylabel('k66');

    subplot(3,3,7)
    scatter(X_class(3,:),X_class(4,:),20,y_class,'filled');xlabel('k33');ylabel('k66');
    
end 

