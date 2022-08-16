close all
clc
clear all 
addpath('fsparse')
rng('shuffle')

%Parameters
ConstraintTolerance = 1e-3;
uc = 1; % mm
lx= [300,300];
ly= [75,75];
lz = [25,25];
rmin=sqrt(2);
ncomp = 2;
E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3 
mload = 60e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; %Load
nu = 0.33; % -
penal = 4;
uc = 1; % mm

nelx = 30; %60 
nely = 10; %10 
nelz = 4;
ndofs = 3;

%Bounds
lb = [];
ub = [];
for n=1:ncomp
    [K_rg] = condensationKrg(lx(n),ly(n),lz(n),nelx,nely,nelz,ndofs,E,nu,penal,rmin);
    K_rg = full(K_rg);
    lb = [lb; -0.01*K_rg]; 
    ub = [ub; 1.01*K_rg];
end 


nDVs = 8; % number of variables
file = load("Versuch_7500/Mass_Estimator/ANN_r2_0_98866_mse_0_00088305_samples_8371_lx_300_ly_75_lz_25_nelx_30_nely_10_nelz_4.mat");
ANN = file.net;


file = load("Versuch_7500/Feasibility_Estimator/SVM_acc_0_92032_false_pos_0_028796_true_pos_0_90302_C_5_samples_7530_5622_lx_300_ly_75_lz_25_nelx_30_nely_10_nelz_4.mat");
SVM = file.SVM;

file = load('Versuch_7500/Parameter_ScoreSVM.mat');
ScoreSVM = file.ScoreSVM;
% ScoreSVM = fitSVMPosterior(SVM);

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
    
    D = K(1:6,1:6)\[0;Fload;0;0;0;0] % displacement

    max(0,D(2)/uc - 1 - ConstraintTolerance)
    
%       D = K(4:9,4:9)\[0;0;0;0;0;Fload*100000] % displacement
% 
%       max(0,D(6)/uc - 1 - ConstraintTolerance)

%     D = K(4:9,4:9)\[0;0;0;0;Fload;0] % displacement
% 
%     max(0,D(5)/uc - 1 - ConstraintTolerance)


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
        
%         D = K(1:6,1:6)\[0;Fload;0;0;0;0]; % displacement
% 
%         y3 = max(0,abs(D(2))/uc - 1 - ConstraintTolerance);

        D = K(4:9,4:9)\[0;0;0;Fload;0;0];
        y3 = max(0,abs(D(4))/uc - 1 - ConstraintTolerance); % - -> +
        
%         D = K(4:9,4:9)\[0;0;0;0;Fload;0];
%         y3 = max(0,abs(D(5))/uc - 1 - ConstraintTolerance); % - -> +
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


function K = stiffnessMatrix_brick (E,nu,length_x,length_y,length_z)
% STIFFNESSMATRIX_BRICK Compute stiffness matrix for brick element
%   K = stiffnessMatrix_brick (E,nu,length_x,length_y,length_z) Computes
%   the 24x24 stiffness matrix for a regular 8 noded hexahedral finite 
%   element with Young´s modulus "E", Poisson ratio "nu" and lengths in x, 
%   y and z direction "length_x", "length_y" and "length_z" respectively.
%   Numerical integration is performed with 8 Gauss points, which yields
%   exact results for regular elements. Weight factors are one and 
%   therefore not included in the code.
%
%   Contact: Diego.Petraroia@rub.de
%
    % Compute 3D constitutive matrix (linear continuum mechanics)
    C = E./((1+nu)*(1-2*nu))*[1-nu nu nu 0 0 0; nu 1-nu nu 0 0 0;...
        nu nu 1-nu 0 0 0; 0 0 0 (1-2*nu)/2 0 0; 0 0 0 0 (1-2*nu)/2 0;...
        0 0 0 0 0 (1-2*nu)/2];
    %
    % Gauss points coordinates on each direction
    GaussPoint = [-1/sqrt(3), 1/sqrt(3)];
    %
    % Matrix of vertices coordinates. Generic element centred at the origin.
    coordinates = zeros(8,3);
    coordinates(1,:) = [-length_x/2 -length_y/2 -length_z/2];
    coordinates(2,:) = [length_x/2 -length_y/2 -length_z/2];
    coordinates(3,:) = [length_x/2 length_y/2 -length_z/2];
    coordinates(4,:) = [-length_x/2 length_y/2 -length_z/2];
    coordinates(5,:) = [-length_x/2 -length_y/2 length_z/2];
    coordinates(6,:) = [length_x/2 -length_y/2 length_z/2];
    coordinates(7,:) = [length_x/2 length_y/2 length_z/2];
    coordinates(8,:) = [-length_x/2 length_y/2 length_z/2];

    %
    % Preallocate memory for stiffness matrix
    K = zeros (24,24);
    % Loop over each Gauss point
    for xi1=GaussPoint
        for xi2=GaussPoint
            for xi3=GaussPoint
                % Compute shape functions derivatives
                dShape = (1/8)*[-(1-xi2)*(1-xi3),(1-xi2)*(1-xi3),...
                    (1+xi2)*(1-xi3),-(1+xi2)*(1-xi3),-(1-xi2)*(1+xi3),...
                    (1-xi2)*(1+xi3),(1+xi2)*(1+xi3),-(1+xi2)*(1+xi3);...
                    -(1-xi1)*(1-xi3),-(1+xi1)*(1-xi3),(1+xi1)*(1-xi3),...
                    (1-xi1)*(1-xi3),-(1-xi1)*(1+xi3),-(1+xi1)*(1+xi3),...
                    (1+xi1)*(1+xi3),(1-xi1)*(1+xi3);-(1-xi1)*(1-xi2),...
                    -(1+xi1)*(1-xi2),-(1+xi1)*(1+xi2),-(1-xi1)*(1+xi2),...
                    (1-xi1)*(1-xi2),(1+xi1)*(1-xi2),(1+xi1)*(1+xi2),...
                    (1-xi1)*(1+xi2)];
                % Compute Jacobian matrix
                JacobianMatrix = dShape*coordinates;
                % Compute auxiliar matrix for construction of B-Operator
                auxiliar = inv(JacobianMatrix)*dShape;
                % Preallocate memory for B-Operator
                B = zeros(6,24);
                % Construct first three rows
                for i=1:3
                    for j=0:7
                        B(i,3*j+1+(i-1)) = auxiliar(i,j+1);
                    end
                end
                % Construct fourth row
                for j=0:7
                    B(4,3*j+1) = auxiliar(2,j+1);
                end
                for j=0:7
                    B(4,3*j+2) = auxiliar(1,j+1);
                end
                % Construct fifth row
                for j=0:7
                    B(5,3*j+3) = auxiliar(2,j+1);
                end
                for j=0:7
                    B(5,3*j+2) = auxiliar(3,j+1);
                end
                % Construct sixth row
                for j=0:7
                    B(6,3*j+1) = auxiliar(3,j+1);
                end
                for j=0:7
                    B(6,3*j+3) = auxiliar(1,j+1);
                end

                % Add to stiffness matrix
                K = K + B'*C*B*det(JacobianMatrix);
            end
        end
    end
end

function [K_rg] = condensationKrg(lx,ly,lz,nelx,nely,nelz,ndofs,E,nu,penal,rmin)

    
    a = 0.49*lx/nelx; %x
    b = 0.49*ly/nely; %y
    c = 0.49*lz/nelz; %z

    if mod(nelx,2) ~= 0 || mod(nely,2) ~= 0
        fprintf('Must have even element number! \n')
        return
    end 

    %Coordinates of the 3d elements
    coordx = -2*a*nelx/2:2*a:2*a*nelx/2; 
    coordy = 2*b*nely/2:-2*b:-2*b*nely/2;
    coordz = -2*c*nelz/2:2*c:2*c*nelz/2;
    [coordX,coordY,coordZ] = meshgrid(coordx,coordy,coordz);


    KE0 = stiffnessMatrix_brick(E, nu, 2*a, 2*b, 2*c); %full element stiffness matrix
    KE = KE0(tril(ones(length(KE0)))==1); % vector of lower triangular element stiffness matrix
    
    % Prepare filter
    bcF = 0; % zero-Dirichlet BC
    % bcF = 'symmetric'; % zero-Neumann BC
    [dy,dz,dx]=meshgrid(-ceil(rmin)+1:ceil(rmin)-1,...
        -ceil(rmin)+1:ceil(rmin)-1,-ceil(rmin)+1:ceil(rmin)-1 );
    h = max( 0, rmin - sqrt( dx.^2 + dy.^2 + dz.^2 ) );                          % conv. kernel                #3D#
    sH = imfilter( ones( nely, nelz , nelx-2), h, bcF );                         % matrix of weights (filter)  #3D#
    dHs = sH;


    %Prepare Assembly of Stiffness Matrix
    nEl = nelx * nely * nelz;                                                  % number of elements          #3D#
    nodenrs = int32( reshape( 1 : ( 1 + nelx ) * ( 1 + nely ) * ( 1 + nelz ), ...
        1 + nely, 1 + nelz, 1 + nelx ) );                                      % nodes numbering             #3D#
    edofVec = reshape( 3 * nodenrs( 1 : nely, 1 : nelz, 1 : nelx ) + 1, nEl, 1 ); %                             #3D#
    edofMat = edofVec+int32( [0,1,2,3*(nely+1)*(nelz+1)+[0,1,2,-3,-2,-1],-3,-2,-1,3*(nely+...
       1)+[0,1,2],3*(nely+1)*(nelz+2)+[0,1,2,-3,-2,-1],3*(nely+1)+[-3,-2,-1]]);% connectivity matrix         #3D#
    nDof = ( 1 + nely ) * ( 1 + nelz ) * ( 1 + nelx ) * 3;                     % total number of DOFs        #3D#
    [ sI, sII ] = deal( [ ] );
    for j = 1 : 24
        sI = cat( 2, sI, j : 24 );
        sII = cat( 2, sII, repmat( j, 1, 24 - j + 1 ) );
    end
    [ iK , jK ] = deal( edofMat( :,  sI )', edofMat( :, sII )' );
    Iar = sort( [ iK( : ), jK( : ) ], 2, 'descend' ); clear iK jK              % reduced assembly indexing

    %% Stiffness Matrix
    %3d global element stiffness matrix 
    
    x = ones(nely,nelz,nelx);
    
    sK = reshape(KE(:)*(x(:)'.^penal),length(KE)*nelx*nely*nelz,1); 
    K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
    K = K + K' - diag( diag( K ) );
    % Guyan
    K_g = fsparse(2*3*(nely+1)*(nelz+1)+2*ndofs,2*3*(nely+1)*(nelz+1)+2*ndofs,0);
    [row,col]= size(K);
    K_ = sparse(row+ndofs*2,col+ndofs*2);
    K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
    alldofs0_g   = [1:length(K_)];
    mdofs_g = [1:(3*(nely+1)*(nelz+1))+ndofs,length(K_)-(3*(nely+1)*(nelz+1))+1-ndofs:length(K_)];
    sdofs_g = setdiff(alldofs0_g,mdofs_g);
    alldofs_g = [mdofs_g, sdofs_g]; 
    Kss = K_(sdofs_g,sdofs_g);
    Ksm = K_(sdofs_g,mdofs_g);
    T_g = [speye(length(mdofs_g)); -Kss\Ksm];
    T_gt = transpose(T_g);

   
    % RBE
    alldofs0_r = 1:length(K_g);                     %All dofs in original order
    sdofs_r = [ndofs+1:length(K_g)-ndofs];                    %Dofs that are to be removed
    mdofs_r = setdiff(alldofs0_r,sdofs_r);            %Dofs that remain
    alldofs_r = [mdofs_r,sdofs_r];                    %New order, sdofs are at the end
    newdofs_r = zeros(length(alldofs_r),1);           %Nonzeros will remove the condensed nodes
    newdofs_r(mdofs_r) =  1:length(mdofs_r);          %Accesing the new order with the old one
    newdofs_r(ndofs+1:end-ndofs) = 2*ndofs+1:length(mdofs_g); 


    %Coordinates of the free nodes 
    coordRBE = [-lx/2,lx/2;
                0,0;
                0,0]; 

    C = fsparse(length(sdofs_r),length(K_g),0);   
    %% Rigid Body Left Side
    idx = 1;
    for n = 1:(nely+1)*(nelz+1)
        C(3*(n-1)+1,1) =1;                                                      % First DOF of independent node
        C(3*(n-1)+2,2) =1;                                                      % Second DOF of independent node

        C_tz = cross([0;0;1],[coordX(1,1,1) - coordRBE(1,1); coordY(idx,1,1); 0]);
        C(3*(n-1)+1,3) =C_tz(1);                                                 % Third DOF of free node
        C(3*(n-1)+2,3) = C_tz(2);                                                % Third DOF of free node
        C(3*(n-1)+3,3) = C_tz(3);                                                % Third DOF of free node

        C(3*(n-1)+1,4+(n-1)*3) = -1;                                           % Slave nodes of 3d elements to be removed
        C(3*(n-1)+2,4+(n-1)*3+1) = -1;                                         % Slave nodes of 3d elements to be removed
        C(3*(n-1)+3,4+(n-1)*3+2) = -1;                                         % Slave nodes of 3d elements to be removed

        if mod(idx,(nely+1)) == 0
            idx = 1;
        else
            idx = idx+1;
        end
    end 
    %% Rigid Body Right Side
    for n = 1:(nely+1)*(nelz+1)
        
        C(3*(n-1)+1+(nely+1)*(nelz+1)*3,2*3*(nely+1)*(nelz+1)+4) =1;                            % First DOF of independent node
        C(3*(n-1)+2+(nely+1)*(nelz+1)*3,2*3*(nely+1)*(nelz+1)+5) =1;                            % Second DOF of independent node
    
        C_tz = cross([0;0;1],[coordX(1,end,1) - coordRBE(1,2); coordY(idx,1,1); 0]);            % Third DOF of independent node
        C(3*(n-1)+1+(nely+1)*(nelz+1)*3,2*3*(nely+1)*(nelz+1)+6) =C_tz(1);                      % Third DOF of independent node
        C(3*(n-1)+2+(nely+1)*(nelz+1)*3,2*3*(nely+1)*(nelz+1)+6) = C_tz(2);                     % Third DOF of independent node
        C(3*(n-1)+3+(nely+1)*(nelz+1)*33,2*3*(nely+1)*(nelz+1)+6) = C_tz(3);    
     
        C(3*(n-1)+1+(nely+1)*(nelz+1)*3,end  -3 - 3*(nely+1)*(nelz+1) + 3*(n-1)+1) =-1;          % Dependent node of 3d elements to be removed
        C(3*(n-1)+2+(nely+1)*(nelz+1)*3,end  -3 - 3*(nely+1)*(nelz+1) + 3*(n-1)+2) = -1;         % Dependent node of 3d elements to be removed
        C(3*(n-1)+3+(nely+1)*(nelz+1)*3,end  -3 - 3*(nely+1)*(nelz+1) + 3*(n-1)+3) = -1;         % Dependent node of 3d elements to be removed
    
        if mod(idx,(nely+1)) == 0
            idx = 1;
        else
            idx = idx+1;
        end
    end


    %Set up model for the unconstrained case
    Tsm = -C(:,sdofs_r)\C(:,mdofs_r);
    Ti = speye(length(mdofs_r)); 
    T_r = [Ti;Tsm];
    T_rt = transpose(T_r);
    T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
    T_rg =  T_g*T_r(newdofs_r,1:end);
    K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;   
    K_rg  = [K_rg(1,1); K_rg(2,2); K_rg(3,3); K_rg(6,6)];
end 
    
    


