clear all 
close all
clc
addpath('fsparse')
rng('shuffle')


file = load("Ergebnis1117/Versuch1_Parameter_results_X");
K_0 = file.results_X(1:4,1)';
% K_0 = [1.0031e+04, 418.4683, 2.6274e+07, 2.2549e+07]; %K1
% K_0 = [1.8983e+03, 197.0604, 1.7008e+07, 4.9670e+05]; %K2
% K_0(4) = K_0(4) + K_0(4)*0.5;

%Optimization
rmin=1.1;

%Parameters
E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3 
mload = 5e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; %Load
nu = 0.33; % -
penal = 3;
uc = 1; % mm

%Number of elements
lx=300;
ly=100;
nelx = 30; %60
nely = 10; %10 
a = 0.49*lx/nelx; %x
b = 0.49*ly/nely; %y
h = 1;

if mod(nelx,2) ~= 0 || mod(nely,2) ~= 0
    fprintf('Must have even element number! \n')
    return
end 

%Coordinates of the 2d elements
coordx = -2*a*nelx/2:2*a:2*a*nelx/2; 
coordy = -2*b*nely/2:2*b:2*b*nely/2;
[coordX,coordY] = meshgrid(coordx,coordy);


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

%% Left Side
for n = 1:nely+1
    C(2*(n-1)+1,1) =1;                                                      % First DOF of independent node
    C(2*(n-1)+2,2) =1;                                                      % Second DOF of independent node
    
    C_t = cross([0;0;1],[coordX(n,1) - coordRBE(1,1); coordY(n,1);0]);      % Third DOF of independent node
    C(2*(n-1)+1,3) =C_t(1);                                                 % Third DOF of independent node
    C(2*(n-1)+2,3) = C_t(2);                                                % Third DOF of independent node
    
    C(2*(n-1)+1,3+(n-1)*2+1) =-1;                                           % Dependent node of 2d elements to be removed
    C(2*(n-1)+2,3+n*2) = -1;                                                % Dependent node of 2d elements to be removed
end 
%% Right Side
for n = 1:nely+1
    C(2*(n-1)+1+(nely+1)*2,4*(nely+1)+4) =1;                                % First DOF of independent node
    C(2*(n-1)+2+(nely+1)*2,4*(nely+1)+5) =1;                                % Second DOF of independent node
    
    C_t = cross([0;0;1],[coordX(n,end) - coordRBE(1,2); coordY(n,end);0]);  % Third DOF of independent node
    C(2*(n-1)+1+(nely+1)*2,4*(nely+1)+6) =C_t(1);                           % Third DOF of independent node
    C(2*(n-1)+2+(nely+1)*2,4*(nely+1)+6) = C_t(2);                          % Third DOF of independent node
    
    C(2*(n-1)+1+(nely+1)*2,(nely+1)*2+3 + 2*(n-1)+1) =-1;                   % Dependent node of 2d elements to be removed
    C(2*(n-1)+2+(nely+1)*2,(nely+1)*2+3 + 2*(n-1)+2) = -1;                  % Dependent node of 2d elements to be removed
end 

Q = fsparse(size(C,1),1,0);                       %Quadratic Matrix 
%Set up model for the unconstrained case
Tsm = -C(:,sdofs_r)\C(:,mdofs_r);
Ti = speye(length(mdofs_r));
T_r = [Ti;Tsm]; 
T_rt = transpose(T_r);


Kmax = zeros(2,1);
Kmin = 1e16*ones(2,1);

tic
Iter = 200;
count_opt = zeros(Iter,1);
count_feas = zeros(Iter,1);
parfor ii = 1:Iter

    x = ones(nely,nelx);
    x(1:nely/2,1:nelx)= rand(nely/2,nelx);x(nely/2+1:end,1:nelx) = x(linspace(nely/2,1,nely/2),1:nelx); %% Symmetry

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

    % K_21, K_31 == 0   
    %     K_rg(1,1)
    %     K_rg(2,1)
    %     K_rg(2,2)
    %     K_rg(3,1)
    %     K_rg(3,3)
    %     K_rg(6,6)

    
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
    T_gt = transpose(T_g);
    T_rg = T_g*T_r(newdofs_r,1:end);
    T_rgt = transpose(T_rg); 
    K_g = T_gt*K_(alldofs_g,alldofs_g)*T_g;
    K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;

    % GRADIENTS
    dK_rg = zeros(nely*(nelx-2),mm/2); el = 0;
    for elx = 2:nelx-1
      for ely = 1:nely/2
        el = el+1;
        dK_ = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,2);
        n1 = (nely+1)*(elx-1)+ely; 
        n2 = (nely+1)* elx   +ely;
        edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
        dK_(edof+3,edof+3) =  penal*x(ely,elx)^(penal-1)*KE;
        dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg;   
        dK_rg_t = dK_rg_t([1,8,15,36]);
        dK_rg(el,:) = dK_rg_t./K_0;
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
        T_gt = transpose(T_g);
        T_rg = T_g*T_r(newdofs_r,1:end);
        T_rgt = transpose(T_rg); 
        K_g = T_gt*K_(alldofs_g,alldofs_g)*T_g;
        K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;

        % GRADIENTS
        dK_rg = zeros(nely*(nelx-2),mm/2); el = 0;
        for elx = 2:nelx-1
          for ely = 1:nely/2
            el = el+1;
            dK_ = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,2);
            n1 = (nely+1)*(elx-1)+ely; 
            n2 = (nely+1)* elx   +ely;
            edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
            dK_(edof+3,edof+3) =  penal*x(ely,elx)^(penal-1)*KE;
            dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg; 
            dK_rg_t = dK_rg_t([1,8,15,36]);
            dK_rg(el,:) = dK_rg_t./K_0;
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



        % Convergence Check
        change_x = max(abs(xval-xold1));
        feasible_f = max(k);

        if (feasible_f < ConstraintTolerance) && (feasibleflag ~= 1)
            feasibleflag = 1;
            count_feas(ii) = 1;
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
        elseif (change_x < StepTolerance) && feasible_f > ConstraintTolerance  && feasibleflag ~=1
            conv=1;
            exitflag =-2; 
        % If the steptolerance is below the limit and the design is
        % feasible: Mass Sample
        elseif (change_x < StepTolerance) && feasible_f < ConstraintTolerance 
            conv=1;
            exitflag =2;
            count_opt(ii) = 1;
        end


    end

end 

toc

sum(count_opt)
sum(count_feas)




