clear all 
close all
clc
addpath('fsparse')
rng('shuffle')
dbstop if error; 


%% PARAMETRIZATION
rmin=sqrt(2);

%Parameters
E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3 
mload = 60e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; %Load
nu = 0.33; % -
penal = 4; % Penalty

%Number of elements
nelx = 30; 
nely = 10;
nelz = 4;
ndofs = 3;


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
Iar1 = Iar(:,1);
Iar2 = Iar(:,2);


%Stiffness Bounds
lx=300; %mm
ly=75; %mm
lz=25; %mm

 
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

coordxEl = -2*a*nelx/2+a:2*a:2*a*nelx/2-a;  
coordyEl = 2*b*nely/2-b:-2*b:-2*b*nely/2+b;
coordzEl = -2*c*nelz/2+c:2*c:2*c*nelz/2-c;
[coordYEl,coordZEl,coordXEl] = meshgrid(coordyEl,coordzEl,coordxEl);



KE0 = stiffnessMatrix_brick(E, nu, 2*a, 2*b, 2*c); %full element stiffness matrix
KE = KE0(tril(ones(length(KE0)))==1); % vector of lower triangular element stiffness matrix


%% Stiffness Matrix
%3d global element stiffness matrix 
x = ones(nely,nelz,nelx);
sK = reshape(KE(:)*(x(:)'.^penal),length(KE)*nelx*nely*nelz,1); 
K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
K = K + K' - diag( diag( K ) );
% Guyan
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
K_g = fsparse(2*3*(nely+1)*(nelz+1)+ndofs*2,2*3*(nely+1)*(nelz+1)+ndofs*2,0);


% RBE
alldofs0_r = 1:length(K_g);                     %All dofs in original order
sdofs_r = [ndofs+1:length(K_g)-ndofs];                    %Dofs that are to be removed
mdofs_r = setdiff(alldofs0_r,sdofs_r);            %Dofs that remain
alldofs_r = [mdofs_r,sdofs_r];
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
Q = fsparse(size(C,1),1,0);  
Tsm = -C(:,sdofs_r)\C(:,mdofs_r);
Ti = speye(length(mdofs_r)); 
T_r = [Ti;Tsm];
T_rt = transpose(T_r);

T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
T_rg =  T_g*T_r(newdofs_r,1:end);
K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;

%Upper and Lower Stiffness Bounds 
x_lb = -0.1*[K_rg(1,1);K_rg(2,2);K_rg(3,3);K_rg(6,6)]; x_lb = full(x_lb);
x_ub = 1.1*[K_rg(1,1);K_rg(2,2);K_rg(3,3);K_rg(6,6)]; x_ub = full(x_ub);


%% Sampling
N_class=[500:500:7500]; %5500
frac = length(N_class)-1;

X = NaN(4,N_class(end));
y = NaN(1,N_class(end));
y_class = -1*ones(1,N_class(end));
X_class = zeros(4,N_class(end));


DoE = lhsdesign(N_class(1),2)';
deltax = 0.1*(1 - 1e-3); %0.75
density = 1e-3 + (1 - 1e-3 - deltax)*linspace(0,1,N_class(1))'.*ones(N_class(1),nely/2*nelz/2*(nelx-2))+ lhsdesign(N_class(1),nely/2*nelz/2*(nelx-2))*deltax;  
% density = [];
% for i = linspace(0.1, 1, 10)
%     deltax = i*(1 - 1e-3); %0.75
%     density = [density; 1e-3 + (1 - 1e-3 - deltax)*linspace(0,1,N_class(1)/10)'.*ones(N_class(1)/10,nely/2*(nelx-2))+ lhsdesign(N_class(1)/10,nely/2*(nelx-2))*deltax];  
% end

% Physical Seed 
parfor i = 1:length(DoE)
        xtemp = density(i,:);
        x = ones(nely,nelz,nelx);
        x(1:nely/2,1:nelz/2,2:nelx-1) =   reshape(xtemp,nely/2,nelz/2,nelx-2);
        x(nely/2+1:end,1:nelz/2,2:nelx-1) = x(linspace(nely/2,1,nely/2),1:nelz/2,2:nelx-1); 
        x(1:nely/2,nelz/2+1:end,2:nelx-1) = x(1:nely/2,linspace(nelz/2,1,nelz/2),2:nelx-1);
        x(nely/2+1:end,nelz/2+1:end,2:nelx-1) = x(linspace(nely/2,1,nely/2),linspace(nelz/2,1,nelz/2),2:nelx-1); 
        
        %INITIAL ANALYSIS
        sK = reshape(KE(:)*(x(:)'.^penal),length(KE)*nelx*nely*nelz,1); 
        K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
        K = K + K' - diag( diag( K ) );
        m = sum(x(:))/(nely*nelx*nelz)*100;

        %Guyan 
        [row,col]= size(K);
        K_ = sparse(row+ndofs*2,col+ndofs*2);
        K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
        Kss = K_(sdofs_g,sdofs_g);
        Kss = K_(sdofs_g,sdofs_g);
        Ksm = K_(sdofs_g,mdofs_g);
        T_g = [speye(length(mdofs_g)); -Kss\Ksm];
        T_gt = transpose(T_g);
        T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
        T_rg =  T_g*T_r(newdofs_r,1:end);
        K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;   
        X_class(:,i) = [K_rg(1,1);K_rg(2,2);K_rg(3,3);K_rg(6,6)]; 
        y_class(:,i) = 1;
end 

plotActiveLearning(X_class,y_class,100)  

%%
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

%%
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
        StepTolerance = 1e-4;
        x = ones(nely,nelz,nelx);
        x(:,:,2:nelx-1) = 0.5*ones(nely,nelz,(nelx-2));  
        loop = 0; 
        xold1 =  reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);      % For the MMA-Algorithm
        xold2 =  reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]); 
        mm = 8;                                         % Number of constraints
        nn=nelx*nely*nelz - 2*nely*nelz;                           % Number of designvariables
        aa0=1;                   
        aa=zeros(mm,1);
        cc=1e3*ones(mm,1);
        dd=zeros(mm,1);
        xmin = ones(nn,1)*0.001;         % Lower bounds of design variables
        low = xmin;
        xmax = ones(nn,1);               % Upper bounds of design variables
        upp = xmax;



        %INITIAL ANALYSIS
        sK = reshape(KE(:)*(x(:)'.^penal),length(KE)*nelx*nely*nelz,1); 
        K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
        K = K + K' - diag( diag( K ) );
        m = sum(x(:))/(nely*nelx*nelz)*100;

        %Guyan 
        [row,col]= size(K);
        K_ = sparse(row+ndofs*2,col+ndofs*2);
        K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
        Kss = K_(sdofs_g,sdofs_g);
        Kss = K_(sdofs_g,sdofs_g);
        Ksm = K_(sdofs_g,mdofs_g);
        T_g = [speye(length(mdofs_g)); -Kss\Ksm];
        T_gt = transpose(T_g);
        T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
        T_rg =  T_g*T_r(newdofs_r,1:end);
        K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg; 
        K_rg  = [K_rg(1,1), K_rg(2,2),K_rg(3,3),K_rg(6,6)];

        % GRADIENTS
        dK_rg = fsparse(nely*(nelx-2)*nelz,mm/2,0); el = 0;
        for elx = 2:nelx-1
            for elz = 1:nelz/2
              for ely = 1:nely/2
                  el = el+1;
                  dK_ = fsparse(3*(nelx+1)*(nely+1)*(nelz+1)+2*ndofs, 3*(nelx+1)*(nely+1)*(nelz+1)+2*ndofs,0);
                  edof = edofMat(el+nelz*nely,:)';
                  dK_(edof+ndofs,edof+ndofs) =  penal*x(ely,elz,elx)^(penal-1)*KE0;
                  dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg;   
                  dK_rg(el,:) = dK_rg_t([1,8,15,36]);
              end
              dK_rg(el +1: el + nely/2,:) = dK_rg(linspace(el,el-nely/2+1,(nely/2)),:);
              el = el + nely/2;
            end
            dK_rg(el +1: el + nely*nelz/2,:) = dK_rg(linspace(el,el-nely*nelz/2+1,(nely*nelz/2)),:);
            el = el + nely*nelz/2;
        end
        
        %Filter
        dK_rg = full(dK_rg);
        for ii=1:mm/2
            dK_rg(:,ii) = reshape(imfilter(reshape(dK_rg(:,ii),[nely,nelz,(nelx-2)])./ sH, h , bcF),[(nelx-2)*nely*nelz,1]);
        end
        
        dm = ones(nely,nelz,nelx-2)/(nely*nelx*nelz)*100;
        dm= reshape(dm,[(nelx-2)*nely*nelz,1]);


        %Both sides
        k = zeros(mm,1);
        k(1:mm/2) = (K_rg - K_0)./K_0 - epsilon;
        k(mm/2+1:end) = (K_0 - K_rg)./K_0- epsilon;
        
        dk = zeros((nelx-2)*nely*nelz,mm);
        dk(:,1:mm/2) = [dK_rg(:,1)./K_0(1),dK_rg(:,2)./K_0(2),dK_rg(:,3)./K_0(3),dK_rg(:,4)./K_0(4)];
        dk(:,mm/2+1:end) = -[dK_rg(:,1)./K_0(1),dK_rg(:,2)./K_0(2),dK_rg(:,3)./K_0(3),dK_rg(:,4)./K_0(4)];

        % MMA OPTIMIZATION
        xval =  reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);
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
            x(:,:,2:nelx-1) = reshape(xval,[nely,nelz,nelx-2]);


            %INITIAL ANALYSIS
            sK = reshape(KE(:)*(x(:)'.^penal),length(KE)*nelx*nely*nelz,1); 
            K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
            K = K + K' - diag( diag( K ) );
            % Guyan
            [row,col]= size(K);
            K_ = sparse(row+ndofs*2,col+ndofs*2);
            K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
            Kss = K_(sdofs_g,sdofs_g);
            Kss = K_(sdofs_g,sdofs_g);
            Ksm = K_(sdofs_g,mdofs_g);
            T_g = [speye(length(mdofs_g)); -Kss\Ksm];
            T_gt = transpose(T_g);
            T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
            T_rg =  T_g*T_r(newdofs_r,1:end);
            K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;   
            K_rg  = [K_rg(1,1), K_rg(2,2),K_rg(3,3),K_rg(6,6)];
            m = sum(x(:))/(nely*nelx*nelz)*100;
            
            % GRADIENTS
            dK_rg = fsparse(nely*(nelx-2)*nelz,mm/2,0); el = 0;
            for elx = 2:nelx-1
                for elz = 1:nelz/2
                    for ely = 1:nely/2
                        el = el+1;
                        dK_ = fsparse(3*(nelx+1)*(nely+1)*(nelz+1)+ndofs*2, 3*(nelx+1)*(nely+1)*(nelz+1)+ndofs*2,0);
                        edof = edofMat(el+nelz*nely,:)';
                        dK_(edof+ndofs,edof+ndofs) =  penal*x(ely,elz,elx)^(penal-1)*KE0;
                        dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg;   
                        dK_rg(el,:) = dK_rg_t([1,8,15,36]);
                    end
                    dK_rg(el +1: el + nely/2,:) = dK_rg(linspace(el,el-nely/2+1,(nely/2)),:);
                    el = el + nely/2;
                end
                dK_rg(el +1: el + nely*nelz/2,:) = dK_rg(linspace(el,el-nely*nelz/2+1,(nely*nelz/2)),:);
                el = el + nely*nelz/2;
            end
            
            %Filter
            dK_rg = full(dK_rg);
            for ii=1:mm/2
                dK_rg(:,ii) = reshape(imfilter(reshape(dK_rg(:,ii),[nely,nelz,(nelx-2)])./ sH, h , bcF),[(nelx-2)*nely*nelz,1]);
            end
            
            dm = ones(nely,nelz,nelx-2)/(nely*nelx*nelz)*100;
            dm= reshape(dm,[(nelx-2)*nely*nelz,1]);  

            %Both sides
            k = zeros(mm,1);
            k(1:mm/2) = (K_rg - K_0)./K_0 - epsilon;
            k(mm/2+1:end) = (K_0 - K_rg)./K_0- epsilon;

            dk = zeros((nelx-2)*nely*nelz,mm);
            dk(:,1:mm/2) = [dK_rg(:,1)./K_0(1),dK_rg(:,2)./K_0(2),dK_rg(:,3)./K_0(3),dK_rg(:,4)./K_0(4)];
            dk(:,mm/2+1:end) = -[dK_rg(:,1)./K_0(1),dK_rg(:,2)./K_0(2),dK_rg(:,3)./K_0(3),dK_rg(:,4)./K_0(4)];
            
            % MMA OPTIMIZATION
            xval =  reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);
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

    N_temp = round(i^4*(1e6-N_temp_0 )/(frac-1)^4 +  N_temp_0); %^2 || 1e7
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
N= 8000;  %6000
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
    StepTolerance = 1e-4;
    x = ones(nely,nelz,nelx);
    x(:,:,2:nelx-1) = 0.5*ones(nely,nelz,nelx-2);  
    loop = 0; 
    xold1 =  reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);      % For the MMA-Algorithm
    xold2 =  reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);
    mm = 8;                                         % Number of constraints
    nn=nelx*nely*nelz - 2*nely*nelz;                           % Number of designvariables
    aa0=1;                   
    aa=zeros(mm,1);
    cc=1e3*ones(mm,1);
    dd=zeros(mm,1);
    xmin = ones(nn,1)*0.001;         % Lower bounds of design variables
    low = xmin;
    xmax = ones(nn,1);               % Upper bounds of design variables
    upp = xmax;



    %INITIAL ANALYSIS
    sK = reshape(KE(:)*(x(:)'.^penal),length(KE)*nelx*nely*nelz,1); 
    K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
    K = K + K' - diag( diag( K ) );

    %Guyan 
    [row,col]= size(K);
    K_ = sparse(row+ndofs*2,col+ndofs*2);
    K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
    Kss = K_(sdofs_g,sdofs_g);
    Kss = K_(sdofs_g,sdofs_g);
    Ksm = K_(sdofs_g,mdofs_g);
    T_g = [speye(length(mdofs_g)); -Kss\Ksm];
    T_gt = transpose(T_g);
    T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
    T_rg =  T_g*T_r(newdofs_r,1:end);
    K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;   
    K_rg  = [K_rg(1,1), K_rg(2,2),K_rg(3,3),K_rg(6,6)];
    m = sum(x(:))/(nely*nelx*nelz)*100;

    % GRADIENTS
    dK_rg = fsparse(nely*(nelx-2)*nelz,mm/2,0); el = 0;
    for elx = 2:nelx-1
        for elz = 1:nelz/2
          for ely = 1:nely/2
            el = el+1;
            dK_ = fsparse(3*(nelx+1)*(nely+1)*(nelz+1)+ndofs*2, 3*(nelx+1)*(nely+1)*(nelz+1)+ndofs*2,0);
            edof = edofMat(el+nelz*nely,:)';
            dK_(edof+ndofs,edof+ndofs) =  penal*x(ely,elz,elx)^(penal-1)*KE0;
            dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg;   
            dK_rg(el,:) = dK_rg_t([1,8,15,36]);
          end
          dK_rg(el +1: el + nely/2,:) = dK_rg(linspace(el,el-nely/2+1,(nely/2)),:);
          el = el + nely/2;
        end
        dK_rg(el +1: el + nely*nelz/2,:) = dK_rg(linspace(el,el-nely*nelz/2+1,(nely*nelz/2)),:);
        el = el + nely*nelz/2;
    end 


    %Filter
    dK_rg = full(dK_rg);
    for ii=1:mm/2
        dK_rg(:,ii) = reshape(imfilter(reshape(dK_rg(:,ii),[nely,nelz,(nelx-2)])./ sH, h , bcF),[(nelx-2)*nely*nelz,1]);
    end 

    dm = ones(nely,nelz,nelx-2)/(nely*nelx*nelz)*100;
    dm= reshape(dm,[(nelx-2)*nely*nelz,1]); 


    %Both sides
    k = zeros(mm,1);
    k(1:mm/2) = (K_rg - K_0)./K_0 - epsilon;
    k(mm/2+1:end) = (K_0 - K_rg)./K_0- epsilon;

    dk = zeros((nelx-2)*nely*nelz,mm);
    dk(:,1:mm/2) = [dK_rg(:,1)./K_0(1),dK_rg(:,2)./K_0(2),dK_rg(:,3)./K_0(3),dK_rg(:,4)./K_0(4)];
    dk(:,mm/2+1:end) = -[dK_rg(:,1)./K_0(1),dK_rg(:,2)./K_0(2),dK_rg(:,3)./K_0(3),dK_rg(:,4)./K_0(4)];

    % MMA OPTIMIZATION
    xval =  reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);
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
        x(:,:,2:nelx-1) = reshape(xval,[nely,nelz,nelx-2]);


        %INITIAL ANALYSIS
        sK = reshape(KE(:)*(x(:)'.^penal),length(KE)*nelx*nely*nelz,1); 
        K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
        K = K + K' - diag( diag( K ) );

        %Guyan 
        [row,col]= size(K);
        K_ = sparse(row+ndofs*2,col+ndofs*2);
        K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
        Kss = K_(sdofs_g,sdofs_g);
        Kss = K_(sdofs_g,sdofs_g);
        Ksm = K_(sdofs_g,mdofs_g);
        T_g = [speye(length(mdofs_g)); -Kss\Ksm];
        T_gt = transpose(T_g);
        T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
        T_rg =  T_g*T_r(newdofs_r,1:end);
        K_rg = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;   
        K_rg  = [K_rg(1,1), K_rg(2,2),K_rg(3,3),K_rg(6,6)];
        m = sum(x(:))/(nely*nelx*nelz)*100;

        % GRADIENTS
        dK_rg = fsparse(nely*(nelx-2)*nelz,mm/2,0); el = 0;
        for elx = 2:nelx-1
            for elz = 1:nelz/2
              for ely = 1:nely/2
                el = el+1;
                dK_ = fsparse(3*(nelx+1)*(nely+1)*(nelz+1)+ndofs*2, 3*(nelx+1)*(nely+1)*(nelz+1)+ndofs*2,0);
                edof = edofMat(el+nelz*nely,:)';
                dK_(edof+ndofs,edof+ndofs) =  penal*x(ely,elz,elx)^(penal-1)*KE0;
                dK_rg_t = T_rgt*dK_(alldofs_g,alldofs_g)*T_rg;   
                dK_rg(el,:) = dK_rg_t([1,8,15,36]);
              end
              dK_rg(el +1: el + nely/2,:) = dK_rg(linspace(el,el-nely/2+1,(nely/2)),:);
              el = el + nely/2;
            end
            dK_rg(el +1: el + nely*nelz/2,:) = dK_rg(linspace(el,el-nely*nelz/2+1,(nely*nelz/2)),:);
            el = el + nely*nelz/2;
        end 


        %Filter
        dK_rg = full(dK_rg);
        for ii=1:mm/2
            dK_rg(:,ii) = reshape(imfilter(reshape(dK_rg(:,ii),[nely,nelz,(nelx-2)])./ sH, h , bcF),[(nelx-2)*nely*nelz,1]);
        end 

        dm = ones(nely,nelz,nelx-2)/(nely*nelx*nelz)*100;
        dm= reshape(dm,[(nelx-2)*nely*nelz,1]);   


        %Both sides
        k = zeros(mm,1);
        k(1:mm/2) = (K_rg - K_0)./K_0 - epsilon;
        k(mm/2+1:end) = (K_0 - K_rg)./K_0- epsilon;
    
        dk = zeros((nelx-2)*nely*nelz,mm);
        dk(:,1:mm/2) = [dK_rg(:,1)./K_0(1),dK_rg(:,2)./K_0(2),dK_rg(:,3)./K_0(3),dK_rg(:,4)./K_0(4)];
        dk(:,mm/2+1:end) = -[dK_rg(:,1)./K_0(1),dK_rg(:,2)./K_0(2),dK_rg(:,3)./K_0(3),dK_rg(:,4)./K_0(4)];

        % MMA OPTIMIZATION
        xval =  reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);
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
ratio = 0.05*length(X)/length(X_2);
[idx,~,~] = dividerand(length(X_2),ratio,1-ratio,0);
X_2 = X_2(:,idx);
[idx2,distance2]= knnsearch(X',X_2','Distance','seuclidean');
y_2 = 1.1*y(idx2); y_2(distance2 ==0) = y(distance2 ==0); % Rate 1.2
X_3 =  X_class_2(:,y_class_2(N_mass+1:end) == -1);
[idx3,distance3]= knnsearch(X',X_3','Distance','seuclidean');
y_3 = 1.2*y(idx3);
X_4 = [X,X_2,X_3];
y_4 = [y,y_2,y_3];
 
samples = [X_4;y_4];
save("Versuch_7500\Samples_Mass\samples_" + num2str(length(samples)) + "_lx_" + num2str(lx) + "_ly_" + num2str(ly) + "_lz_" + num2str(lz) + "_nelx_" + num2str(nelx) + "_nely_" + num2str(nely) + "_nelz_" + num2str(nelz) + '_border','samples');

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
save("Versuch_7500\Samples_Classification\samples_" + num2str(length(y_class_4)) + "_lx_" + num2str(lx) + "_ly_"  ...
    + num2str(ly) + "_lz_" + num2str(lz) + "_nelx_" + num2str(nelx) + "_nely_" + num2str(nely) + "_nelz_" + num2str(nelz),'samples_class');
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