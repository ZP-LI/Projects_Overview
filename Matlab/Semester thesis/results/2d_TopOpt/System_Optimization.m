close all
clc
clear all 
addpath('fsparse')
rng('shuffle')

%Parameters
ConstraintTolerance = 1e-3;
uc = 1; % mm
lx= [300,300];
ly= [100,100];
rmin=1.3;
ncomp = 2;
E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3 
mload = 5e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; %Load
nu = 0.33; % -
penal = 3;
% uc = 1; % mm

nelx = 30; %60 
nely = 10; %10 
a = 0.49*lx/nelx; %x
b = 0.49*ly/nely; %y
h = 1;


%Bounds
lb = [];
ub = [];
for n=1:ncomp
    [K_rg]= stiffnessMatrix(nelx,nely,lx(n),ly(n),a,b,h,E,nu,penal);
    lb = [lb; -0.01*K_rg]; 
    ub = [ub; 1.01*K_rg];
end 


nDVs = 8; % number of variables
file = load("Ergebnis1117_14/Mass_Estimator/ANN_r2_0_99349_mse_0_00039472_samples_6119_lx_300_ly_100_nelx_30_nely_10.mat");
ANN = file.net;


file = load("Ergebnis1117_14/Feasibility_Estimator/SVM_acc_0_93044_false_pos_0_049793_true_pos_0_92494_C_5_samples_5531_4328_lx_300_ly_100_nelx_30_nely_10.mat");
SVM = file.SVM;
ScoreSVM = fitSVMPosterior(SVM);

results_X = zeros(8,10);
results_y = zeros(2,10);

%% Multiple Optimization Runs due to stochastical nature of the Particle Swarm Optimization
for i=1:5 %10
    
    costFcn = @(x)objFn(x,lx,ncomp,Fload,uc,ANN,SVM,ScoreSVM,ConstraintTolerance);
    options = optimoptions('particleswarm','SwarmSize',800,'Display','iter','UseParallel',true,'MaxIterations',1600,'SelfAdjustmentWeight',1.63,'SocialAdjustmentWeight', 0.62, 'InertiaRange',[0.657,0.657],'HybridFcn','fmincon'); % 200->800; 400->1600
    [x,fval,exitflag,output] = particleswarm(costFcn,nDVs,lb,ub,options);


    results_y(1,i) = ANN([x(1);x(2);x(3);x(4)]);
    results_X(1,i) = x(1);
    results_X(2,i) = x(2);
    results_X(3,i) = x(3);
    results_X(4,i) = x(4);

    results_y(2,i) = ANN([x(5);x(6);x(7);x(8)])
    results_X(5,i) = x(5);
    results_X(6,i) = x(6);
    results_X(7,i) = x(7);
    results_X(8,i) = x(8)

    results_y(1,i) + results_y(2,i)

   
    Kc = zeros(6,6,2);
    feasible = zeros(ncomp,1);
    post = zeros(ncomp,1);
    
    for n = 1:ncomp
        
        K_cs1_1 = x(1 + (n-1)*4);
        K_cs2_2 = x(2 + (n-1)*4);
        K_cs3_3 = x(3 + (n-1)*4);
        K_cs6_6 = x(4 + (n-1)*4);
        K_cs2_1 = 0;
        K_cs3_1 = 0;

        Kc(:,:,n) = ...
        [              K_cs1_1,                                   K_cs2_1,                                    K_cs3_1,             -K_cs1_1,                                   -K_cs2_1,                       K_cs2_1*lx(n) - K_cs3_1
            K_cs2_1,                                   K_cs2_2,  (K_cs2_2*lx(n)^2 + K_cs3_3 - K_cs6_6)/(2*lx(n)),             -K_cs2_1,                                   -K_cs2_2,  (K_cs2_2*lx(n)^2 - K_cs3_3 + K_cs6_6)/(2*lx(n))
              K_cs3_1, (K_cs2_2*lx(n)^2 + K_cs3_3 - K_cs6_6)/(2*lx(n)),                                    K_cs3_3,             -K_cs3_1, -(K_cs2_2*lx(n)^2 + K_cs3_3 - K_cs6_6)/(2*lx(n)),   (K_cs2_2*lx(n)^2)/2 - K_cs3_3/2 - K_cs6_6/2
             -K_cs1_1,                                  -K_cs2_1,                                   -K_cs3_1,              K_cs1_1,                                    K_cs2_1,                       K_cs3_1 - K_cs2_1*lx(n)
             -K_cs2_1,                                  -K_cs2_2, -(K_cs2_2*lx(n)^2 + K_cs3_3 - K_cs6_6)/(2*lx(n)),              K_cs2_1,                                    K_cs2_2, -(K_cs2_2*lx(n)^2 - K_cs3_3 + K_cs6_6)/(2*lx(n))
 K_cs2_1*lx(n) - K_cs3_1, (K_cs2_2*lx(n)^2 - K_cs3_3 + K_cs6_6)/(2*lx(n)),   (K_cs2_2*lx(n)^2)/2 - K_cs3_3/2 - K_cs6_6/2, K_cs3_1 - K_cs2_1*lx(n), -(K_cs2_2*lx(n)^2 - K_cs3_3 + K_cs6_6)/(2*lx(n)),                                    K_cs6_6];
 
        [Phi,Lambda] = eig(Kc(:,:,n));  % lambda eigenvalue
        Lambda(abs(Lambda) < 1e-3) = 0; 
        [Label,~] = predict(SVM,[x(1+4*(n-1)),x(2+4*(n-1)),x(3+4*(n-1)),x(4+4*(n-1))]);
        [~,PostProbs]= predict(ScoreSVM,[x(1+4*(n-1)),x(2+4*(n-1)),x(3+4*(n-1)),x(4+4*(n-1))]);
        feasible(n) = Label;
        post(n) = PostProbs(2);
        if any(Lambda(:) < 0)
            feasible(n) = -1;
            post(n) = 0;
        end 
    end 
    
    
    K = zeros((ncomp+1)*3, (ncomp+1)*3); % zeros(9,9)
    for elx = 1:ncomp
        n1 =  3*(elx-1)+1; 
        n2 =  3*elx+1;
	    edof = [n1;n1+1;n1+2; n2;n2+1;n2+2];
        K(edof,edof) = K(edof,edof) + Kc(:,:,elx);
    end

    D = K(4:9,4:9)\[0;0;0;0;Fload;0] % displacement
    
    max(0,D(5)/uc - 1 - ConstraintTolerance)


end



%% 1             
function y =objFn(x,lx,ncomp,Fload,uc,ANN,SVM,ScoreSVM,ConstraintTolerance)

%x = 
    Kc = zeros(6,6,2);
    feasible = zeros(ncomp,1);
    post = zeros(ncomp,1);
    
    for n = 1:ncomp
        
        K_cs1_1 = x(1 + (n-1)*4);
        K_cs2_2 = x(2 + (n-1)*4);
        K_cs3_3 = x(3 + (n-1)*4);
        K_cs6_6 = x(4 + (n-1)*4);
        K_cs2_1 = 0;
        K_cs3_1 = 0;

        Kc(:,:,n) = ...
        [              K_cs1_1,                                   K_cs2_1,                                    K_cs3_1,             -K_cs1_1,                                   -K_cs2_1,                       K_cs2_1*lx(n) - K_cs3_1
            K_cs2_1,                                   K_cs2_2,  (K_cs2_2*lx(n)^2 + K_cs3_3 - K_cs6_6)/(2*lx(n)),             -K_cs2_1,                                   -K_cs2_2,  (K_cs2_2*lx(n)^2 - K_cs3_3 + K_cs6_6)/(2*lx(n))
              K_cs3_1, (K_cs2_2*lx(n)^2 + K_cs3_3 - K_cs6_6)/(2*lx(n)),                                    K_cs3_3,             -K_cs3_1, -(K_cs2_2*lx(n)^2 + K_cs3_3 - K_cs6_6)/(2*lx(n)),   (K_cs2_2*lx(n)^2)/2 - K_cs3_3/2 - K_cs6_6/2
             -K_cs1_1,                                  -K_cs2_1,                                   -K_cs3_1,              K_cs1_1,                                    K_cs2_1,                       K_cs3_1 - K_cs2_1*lx(n)
             -K_cs2_1,                                  -K_cs2_2, -(K_cs2_2*lx(n)^2 + K_cs3_3 - K_cs6_6)/(2*lx(n)),              K_cs2_1,                                    K_cs2_2, -(K_cs2_2*lx(n)^2 - K_cs3_3 + K_cs6_6)/(2*lx(n))
              K_cs2_1*lx(n) - K_cs3_1, (K_cs2_2*lx(n)^2 - K_cs3_3 + K_cs6_6)/(2*lx(n)),   (K_cs2_2*lx(n)^2)/2 - K_cs3_3/2 - K_cs6_6/2, K_cs3_1 - K_cs2_1*lx(n), -(K_cs2_2*lx(n)^2 - K_cs3_3 + K_cs6_6)/(2*lx(n)),                                    K_cs6_6];
 

        [Phi,Lambda] = eig(Kc(:,:,n));
        Lambda(abs(Lambda) < 1e-3) = 0; 
        [Label,~] = predict(SVM,[x(1+4*(n-1)),x(2+4*(n-1)),x(3+4*(n-1)),x(4+4*(n-1))]);
        [~,PostProbs]= predict(ScoreSVM,[x(1+4*(n-1)),x(2+4*(n-1)),x(3+4*(n-1)),x(4+4*(n-1))]);
        feasible(n) = Label;
        post(n) = PostProbs(2);
        if any(Lambda(:) < 0)
            feasible(n) = -1;
            post(n) = 0;
        end 
    end 
    
    
    K = zeros((ncomp+1)*3, (ncomp+1)*3);
    for elx = 1:ncomp
        n1 =  3*(elx-1)+1; 
        n2 =  3*elx+1;
        edof = [n1;n1+1;n1+2; n2; n2+1;n2+2];
        K(edof,edof) = K(edof,edof) + Kc(:,:,elx);
    end
    
    %Mass [0,100] + [0,100] 
    y1 = max(0,ANN([x(1);x(2);x(3);x(4)]));
    y2 = max(0,ANN([x(5);x(6);x(7);x(8)]));
    
    
    % Displacement [0,1+]
    y3 = 1;
    if rcond(K(4:9,4:9)) > 1e-12 %returns an estimate for the reciprocal condition of A in 1-norm. If A is well conditioned, rcond(A) is near 1.0. If A is badly conditioned, rcond(A) is near 0.
        % Displacement
        D = K(4:9,4:9)\[0;0;0;0;Fload;0];
        y3 = max(0,abs(D(5))/uc -1 - ConstraintTolerance); 
    end
    
    % Feasibility [0,1]
    y4 = 1- post(1); 
    if feasible(1) == 1 
        y4=0;
    end
    y5 = 1- post(2); 
    if feasible(2) == 1 
        y5=0;
    end

    %Objective Function 
    % --> 2*100 because maximum mass of system
    % --> 200/ConstraintTolerance, because slightest distance from feasible area that should be higher than maximum weight 
    y = y1 + y2 + 2000/ConstraintTolerance*(y3+y4+y5); %200->2000
               
end 


function [K_rg] = stiffnessMatrix(nelx,nely,lx,ly,a,b,h,E,nu,penal)
    
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
    
    
    %% Guyan
    x = ones(nely,nelx);
    sK = reshape(KE(:)*(x(:)'.^penal),64*nelx*nely,1);
    K = fsparse(iK,jK,sK); K = (K+K')/2;
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
    Kss = K_(sdofs_g,sdofs_g);
    Ksm = K_(sdofs_g,mdofs_g);
    T_g = [speye(length(mdofs_g)); -Kss\Ksm];
    T_rg = T_g*T_r(newdofs_r,1:end);
    T_rgt = transpose(T_rg); 
    K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;
    K_rg = [K_rg(1,1);K_rg(2,2);K_rg(3,3);K_rg(6,6)]; 
end 
    
    


