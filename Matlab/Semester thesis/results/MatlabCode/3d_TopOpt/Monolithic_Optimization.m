close all
clc
clear 
addpath('fsparse')
rng('default')
dbstop if error; 


uc=1; %
ncomp = 2;

%Optimization
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
Q0 = [fsparse(length(mdofs_r),1,0);C(:,sdofs_r)\Q]; 



% INITIALIZE OPTIMIZATION
MaxIterations= 1000;
ConstraintTolerance = 1e-3;
StepTolerance = 1e-4;
x = ones(nely,nelz,nelx*ncomp);
x(:,:,2:nelx-1) = 0.5*ones(nely, nelz, (nelx-2)); 
x(:,:,nelx+2:2*nelx-1) = 0.5*ones(nely, nelz, (nelx-2)); 
xold1 =  [reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1])];      % For the MMA-Algorithm
xold2 =  [reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1])]; 
mm = 1;                                         % Number of constraints
nn=nelx*nely*nelz*ncomp - 2*nely*nelz*ncomp;                           % Number of designvariables
aa0=1;                   
aa=zeros(mm,1);
cc=1e3*ones(mm,1);
dd=zeros(mm,1);
xmin = ones(nn,1)*0.001;         % Lower bounds of design variables
low = xmin;
xmax = ones(nn,1);               % Upper bounds of design variables
upp = xmax;


K_RG = fsparse(ncomp*ndofs*2-ndofs,ncomp*ndofs*2-ndofs,0);
m = 0;
for i=1:ncomp
    %3d global element stiffness matrix 
    xcomp = x(:,:,1 + (i-1)*nelx : nelx + (i-1)*nelx);
    sK = reshape(KE(:)*(xcomp(:)'.^penal),length(KE)*nelx*nely*nelz,1);
    K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
    K = K + K' - diag( diag( K ));
    [row,col]= size(K);
    K_ = sparse(row+ndofs*2,col+ndofs*2);
    K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
    Kss = K_(sdofs_g,sdofs_g);
    Ksm = K_(sdofs_g,mdofs_g);
    T_g = [speye(length(mdofs_g)); -Kss\Ksm];
    T_gt = transpose(T_g);
    T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
    T_rg =  T_g*T_r(newdofs_r,1:end);
    m = m +  sum(xcomp(:))/(nely*nelz*nelx)*100;
    edofs = [1:2*ndofs] + (ndofs)*(i-1);
    K_RG(edofs,edofs) = K_RG(edofs,edofs) + T_rgt*K_(alldofs_g,alldofs_g)*T_rg;
end

U_RG = zeros(length(K_RG),2); 
F_RG = zeros(length(K_RG),2); 
F_RG(end-0,1)=Fload*100000;
F_RG(end-0,2)=1;
freedofs = ndofs+1:length(K_RG);
U_RG(freedofs,:) = K_RG(freedofs,freedofs)\F_RG(freedofs,:);
u = U_RG(end-0,1);


% GRADIENTS
du = zeros(nn,1);
for i=1:ncomp
    %RBE2
    edofs = [1:2*ndofs] + (ndofs)*(i-1);
    U_rg = U_RG(edofs,:);
    U_g = T_r*U_rg + Q0;
    U_g(alldofs_r,:) = U_g;
    %Guyan
    xcomp = x(:,:,1 + (i-1)*nelx : nelx + (i-1)*nelx);
    sK = reshape(KE(:)*(xcomp(:)'.^penal),length(KE)*nelx*nely*nelz,1);
    K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
    K = K + K' - diag( diag( K ));
    [row,col]= size(K);
    K_ = sparse(row+ndofs*2,col+ndofs*2);
    K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
    Kss = K_(sdofs_g,sdofs_g);
    Ksm = K_(sdofs_g,mdofs_g);
    U_ = [U_g;(-Kss\Ksm)*U_g];
    U_(alldofs_g,:) = U_;
    U = U_(ndofs+1:end-ndofs,:);
    el = 0;
    for elx = 2:nelx-1
        for elz = 1:nelz/2
          for ely = 1:nely/2
            el=el+1;
            edof = edofMat(el+nely*nelz,:);
            du(el+nn/2*(i-1)) = -(penal)* xcomp(ely, elz,elx)^(penal-1)*U(edof,2)'*KE0*U(edof,1) ;
          end
          du(el + nn/2*(i-1) +1:el + nn/2*(i-1)  +nely/2) = du(linspace(el + nn/2*(i-1),el + nn/2*(i-1)-nely/2+1,(nely/2)));
          el = el + nely/2;
        end
        du(el +1 + nn/2*(i-1): el + nely*nelz/2 + nn/2*(i-1)) = du(linspace(el + nn/2*(i-1),el-nely*nelz/2+1 + nn/2*(i-1),(nely*nelz/2)));
        el = el + nely*nelz/2;
    end 
    
    %Filter
    du(1 + nely*(nelx-2)*nelz*(i-1):nely*(nelx-2)*nelz+ nely*(nelx-2)*nelz*(i-1)) = reshape(imfilter(reshape(du(1 + nely*(nelx-2)*nelz*(i-1):nely*(nelx-2)*nelz+ nely*(nelx-2)*nelz*(i-1)),[nely,nelz,(nelx-2)])./ sH, h , bcF),[(nelx-2)*nelz*nely,1]);
end 

du = full(du); 
dm = ones(nn,1)/(nely*nelz*nelx)*100;



% MMA OPTIMIZATION
xval =  [reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);...
          reshape(x(:,:,nelx+2:2*nelx-1),[(nelx-2)*nely*nelz,1])];      % For the MMA-Algorithm
f0val =m;     
df0dx= dm; 
df0dx2= 0*df0dx;
fval=u/uc-1;          
dfdx=du';
dfdx2=0*dfdx; 

%     clf; display_3D(x);

conv = 0;
loop = 0; 
%tic
while conv == 0 && loop < MaxIterations
    loop = loop + 1;
    xold = x;
    
    [xmma,ymma,zmma,lam,xsi,eta,mu,zet,ss,low,upp] = ...
        mmasub_old(mm,nn,loop,xval,xmin,xmax,xold1,xold2, ...
        f0val,df0dx,df0dx2,fval,dfdx,dfdx2,low,upp,aa0,aa,cc,dd);
    

    f0valold = f0val;    
    xold2 = xold1;
    xold1 = xval;
    xval = xmma;
    x(:,:,2:nelx-1) = reshape(xval(1:(nelx-2)*nely*nelz),[nely,nelz,nelx-2]);
    x(:,:,nelx+2:nelx*2-1) = reshape(xval((nelx-2)*nely*nelz+1:(nelx-2)*nely*nelz*2),[nely,nelz,nelx-2]);
    
    
    K_RG = fsparse(ncomp*ndofs*2-ndofs,ncomp*ndofs*2-ndofs,0);
    m = 0;
    for i=1:ncomp
        %3d global element stiffness matrix 
        xcomp = x(:,:,1 + (i-1)*nelx : nelx + (i-1)*nelx);
        sK = reshape(KE(:)*(xcomp(:)'.^penal),length(KE)*nelx*nely*nelz,1);
        K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
        K = K + K' - diag( diag( K ));
        [row,col]= size(K);
        K_ = sparse(row+ndofs*2,col+ndofs*2);
        K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
        Kss = K_(sdofs_g,sdofs_g);
        Ksm = K_(sdofs_g,mdofs_g);
        T_g = [speye(length(mdofs_g)); -Kss\Ksm];
        T_gt = transpose(T_g);
        T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
        T_rg =  T_g*T_r(newdofs_r,1:end);
        m = m +  sum(xcomp(:))/(nely*nelz*nelx)*100;
        edofs = [1:2*ndofs] + (ndofs)*(i-1);
        K_RG(edofs,edofs) = K_RG(edofs,edofs) + T_rgt*K_(alldofs_g,alldofs_g)*T_rg;
    end

    U_RG = zeros(length(K_RG),2); 
    F_RG = zeros(length(K_RG),2); 
    F_RG(end-0,1)=Fload*100000;
    F_RG(end-0,2)=1;
    freedofs = ndofs+1:length(K_RG);
    U_RG(freedofs,:) = K_RG(freedofs,freedofs)\F_RG(freedofs,:);
    u = U_RG(end-0,1);


    % GRADIENTS
    du = zeros(nn,1);
    for i=1:ncomp
        %RBE2
        edofs = [1:2*ndofs] + (ndofs)*(i-1);
        U_rg = U_RG(edofs,:);
        U_g = T_r*U_rg + Q0;
        U_g(alldofs_r,:) = U_g;
        %Guyan
        xcomp = x(:,:,1 + (i-1)*nelx : nelx + (i-1)*nelx);
        sK = reshape(KE(:)*(xcomp(:)'.^penal),length(KE)*nelx*nely*nelz,1);
        K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
        K = K + K' - diag( diag( K ));
        [row,col]= size(K);
        K_ = sparse(row+ndofs*2,col+ndofs*2);
        K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
        Kss = K_(sdofs_g,sdofs_g);
        Ksm = K_(sdofs_g,mdofs_g);
        U_ = [U_g;(-Kss\Ksm)*U_g];
        U_(alldofs_g,:) = U_;
        U = U_(ndofs+1:end-ndofs,:);
        el = 0;
        for elx = 2:nelx-1
            for elz = 1:nelz/2
              for ely = 1:nely/2
                el=el+1;
                edof = edofMat(el+nely*nelz,:);
                du(el+nn/2*(i-1)) = -(penal)* xcomp(ely, elz,elx)^(penal-1)*U(edof,2)'*KE0*U(edof,1) ;
              end
              du(el + nn/2*(i-1) +1:el + nn/2*(i-1)  +nely/2) = du(linspace(el + nn/2*(i-1),el + nn/2*(i-1)-nely/2+1,(nely/2)));
              el = el + nely/2;
            end
            du(el +1 + nn/2*(i-1): el + nely*nelz/2 + nn/2*(i-1)) = du(linspace(el + nn/2*(i-1),el-nely*nelz/2+1 + nn/2*(i-1),(nely*nelz/2)));
            el = el + nely*nelz/2;
        end 

        %Filter
        du(1 + nely*(nelx-2)*nelz*(i-1):nely*(nelx-2)*nelz+ nely*(nelx-2)*nelz*(i-1)) = reshape(imfilter(reshape(du(1 + nely*(nelx-2)*nelz*(i-1):nely*(nelx-2)*nelz+ nely*(nelx-2)*nelz*(i-1)),[nely,nelz,(nelx-2)])./ sH, h , bcF),[(nelx-2)*nelz*nely,1]);
    end 

    du = full(du); 
    dm = ones(nn,1)/(nely*nelz*nelx)*100;


    % MMA OPTIMIZATION
    xval =  [reshape(x(:,:,2:nelx-1),[(nelx-2)*nely*nelz,1]);...
              reshape(x(:,:,nelx+2:2*nelx-1),[(nelx-2)*nely*nelz,1])];      % For the MMA-Algorithm
        f0val =m;     
    df0dx= dm; 
    df0dx2= 0*df0dx;
    fval=u/uc-1;          
    dfdx=du';
    dfdx2=0*dfdx; 
    
    
    change_x = max(abs(xval-xold1));
    % Convergence Check
    if (abs(fval)) < ConstraintTolerance && change_x < StepTolerance
        conv = 1;
    end
    
    % PRINT RESULTS
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.3f',m) ...
           'u: ' sprintf('%6.3f',u - uc) ...
            ' ch.: ' sprintf('%6.3f',change_x )])
    
%     %plot
    %clf; display_3D(x,a,b,c);
    if mod(loop,50)==0
        for i=1:ncomp
            figure (i + 100)
            clf;display_3D(x(:,:,1 + (i-1)*nelx:nelx + (i-1)*nelx),a,b,c);
        end 
    end

end 


for i=1:ncomp
    figure (i + 100)
    clf;display_3D(x(:,:,1 + (i-1)*nelx:nelx + (i-1)*nelx),a,b,c);
end 

m=zeros(2,1);
K_rg = zeros(6,6,2);
for i=1:ncomp
    xcomp = x(:,:,1 + (i-1)*nelx : nelx + (i-1)*nelx);
    sK = reshape(KE(:)*(xcomp(:)'.^penal),length(KE)*nelx*nely*nelz,1);
    K = fsparse(Iar(:,1), Iar(:,2), sK, [ nDof, nDof ] );
    K = K + K' - diag( diag( K ));
    [row,col]= size(K);
    K_ = sparse(row+ndofs*2,col+ndofs*2);
    K_(ndofs+1:end-ndofs,ndofs+1:end-ndofs) = K;
    Kss = K_(sdofs_g,sdofs_g);
    Ksm = K_(sdofs_g,mdofs_g);
    T_g = [speye(length(mdofs_g)); -Kss\Ksm];
    T_gt = transpose(T_g);
    T_rgt =  T_rt(1:end,newdofs_r)*T_gt;
    T_rg =  T_g*T_r(newdofs_r,1:end);
    m(i)=  sum(xcomp(:))/(nely*nelz*nelx)*100;
    edofs = [1:2*ndofs] + (ndofs)*(i-1);
    K_rg(:,:,i) = T_rgt*K_(alldofs_g,alldofs_g)*T_rg;
end



K_rg

K_rg(1,1,1)
K_rg(2,2,1)
K_rg(3,3,1)
K_rg(6,6,1)

K_rg(1,1,2)
K_rg(2,2,2)
K_rg(3,3,2)
K_rg(6,6,2)

m

function display_3D(rho,a,b,c)
[nely,nelz,nelx] = size(rho);
hx = 2*a; hy = 2*b; hz = 2*c;            % User-defined unit element size
face = [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8];
set(gcf,'Name','ISO display','NumberTitle','off');
for k = 1:nelx
    x = (k-1)*hx;
    for i = 1:nelz
        z = (i-1)*hz;
        for j = 1:nely
            y = nely*hy - (j-1)*hy;
            if (rho(j,i,k) > 0.1)  % User-defined display density threshold
                vert = [x y z; x y-hy z; x+hx y-hy z; x+hx y z; x y z+hz;x y-hy z+hz; x+hx y-hy z+hz;x+hx y z+hz];
                vert(:,[2 3]) = vert(:,[3 2]); vert(:,2,:) = -vert(:,2,:);
                patch('Faces',face,'Vertices',vert,'FaceColor',[0.3+0.7*(1-rho(j,i,k)),0.3+0.7*(1-rho(j,i,k)),0.3+0.7*(1-rho(j,i,k))],...
                    'FaceAlpha',rho(j,i,k),'EdgeAlpha',rho(j,i,k));
                hold on;
            end
        end
    end
end
    axis equal; axis tight; axis off; box on; view([-30,30]);%view([-120,45]);%view([34,21]); pause(1e-6);
    set(gcf,'renderer','Painters')
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




