clear all
close all
clc
addpath('fsparse')

%% PARAMETRIZATION
rmin=1.1; %Filter radius 
ncomp = 2; %number of components

E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3 
mload = 5e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; %Load
nu = 0.33; % -
penal = 3; %Penalty factor
uc = 1; % mm

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


%% INITIALIZE OPTIMIZATION
MaxIterations= 500;
ConstraintTolerance = 1e-3;
StepTolerance = 1e-3;
x = ones(nely,nelx*ncomp);
x(:,2:nelx-1) = 0.5*ones(nely,(nelx-2)); 
x(:,nelx+2:nelx*ncomp-1) = 0.5*ones(nely,(nelx-2)); 
loop = 0; 
xold1 =  [reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);reshape(x(:,nelx+2:(ncomp*nelx)-1),[(nelx-2)*nely,1])];      % For the MMA-Algorithm
xold2 =  [reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);reshape(x(:,nelx+2:(ncomp*nelx)-1),[(nelx-2)*nely,1])];
mm = 1;                                             % Number of constraints
nn=(nelx*nely - 2*nely)*ncomp;                      % Number of designvariables
aa0=1;                   
aa=zeros(mm,1);
cc=1e3*ones(mm,1);
dd=zeros(mm,1);
xmin = ones(nn,1)*0.001;                            % Lower bounds of design variables
low = xmin;
xmax = ones(nn,1);                                  % Upper bounds of design variables
upp = xmax;


%% Stiffness Matrix 
%Stiffness Matrix of the Design Domain
sK = reshape(KE(:)*(ones(nely*nelx,1)'.^penal),64*nelx*nely,1);
K_ = sparse(iK,jK,sK); 
K_ = (K_+K_')/2; 

%Stiffness Matrix combined with the 3 DOFS for each interface point: 2x3=6
K = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,0);
K(4:end-3,4:end-3) = K_;


%% Only the RBE2 is applied to the design domain, no Guyan Reduction necessary here

alldofs0_r = 1:length(K);                                                   %All dofs in original order
sdofs_r = [4:4+(nely+1)*2-1,length(K)-2-2*(nely+1):length(K)-3];            %Dofs that are to be removed
mdofs_r = setdiff(alldofs0_r,sdofs_r);                                      %Dofs that remain
alldofs_r = [mdofs_r,sdofs_r];                                              %New order, sdofs are at the end

%Coordinates of the free nodes 
coordRBE = [-lx/2,lx/2;
            0,0;
            0,0];

C = fsparse(length(sdofs_r),length(K),0);   
%Left Side 
for n = 1:nely+1
    C(2*(n-1)+1,1) =1;                                                      % First DOF of independent node
    C(2*(n-1)+2,2) =1;                                                      % Second DOF of independent node
    
    C_t = cross([0;0;1],[coordX(n,1) - coordRBE(1,1); coordY(n,1);0]);      % Third DOF of independent node
    C(2*(n-1)+1,3) =C_t(1);                                                 % Third DOF of independent node
    C(2*(n-1)+2,3) = C_t(2);                                                % Third DOF of independent node
    
    C(2*(n-1)+1,3+(n-1)*2+1) =-1;                                           % Dependent node of 2d elements to be removed
    C(2*(n-1)+2,3+n*2) = -1;                                                % Dependent node of 2d elements to be removed
end 
%Right Side
for n = 1:nely+1
    C(2*(n-1)+1+(nely+1)*2,2*(nelx+1)*(nely+1)+4) =1;                                % First DOF of independent node
    C(2*(n-1)+2+(nely+1)*2,2*(nelx+1)*(nely+1)+5) =1;                                % Second DOF of independent node
    
    C_t = cross([0;0;1],[coordX(n,end) - coordRBE(1,2); coordY(n,end);0]);  % Third DOF of independent node
    C(2*(n-1)+1+(nely+1)*2,2*(nelx+1)*(nely+1)+6) =C_t(1);                           % Third DOF of independent node
    C(2*(n-1)+2+(nely+1)*2,2*(nelx+1)*(nely+1)+6) = C_t(2);                          % Third DOF of independent node
    
    C(2*(n-1)+1+(nely+1)*2,(nely+1)*2*nelx+3 + 2*(n-1)+1) =-1;                   % Dependent node of 2d elements to be removed
    C(2*(n-1)+2+(nely+1)*2,(nely+1)*2*nelx+3 + 2*(n-1)+2) = -1;                  % Dependent node of 2d elements to be removed
end 

Q = fsparse(size(C,1),1,0);                       %Quadratic Matrix 
Tsm = -C(:,sdofs_r)\C(:,mdofs_r);
Ti = speye(length(mdofs_r));
T_r = [Ti;Tsm]; 
T_rt = transpose(T_r);
Q0 = [fsparse(length(mdofs_r),1,0);C(:,sdofs_r)\Q];


%% INITIALE ANALYSIS
ndofs = 2*(nelx+1)*(nely+1)-4*(nely+1)+6;
K_R = fsparse(ncomp*ndofs-3,ncomp*ndofs-3,0); %Reduced Global Stiffness Matrix consisting of 2 Components
m=0;
for i=1:ncomp
    sK = reshape(KE(:)*(x(nelx*nely*(i-1)+1:(nelx*nely*i)).^penal),64*nelx*nely,1);
    K_ = sparse(iK,jK,sK); K_ = (K_+K_')/2;
    m = sum(x(:))/(nely*nelx)*100;
    K = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,0);
    K(4:end-3,4:end-3) = K_;
    edofs = [1:length(mdofs_r)] + (length(mdofs_r)-3)*(i-1);
    K_R(edofs,edofs) = K_R(edofs,edofs) + T_rt*K(alldofs_r,alldofs_r)*T_r;
end  

U_R = zeros(length(K_R),2); 
F_R = zeros(length(K_R),2); 
F_R(end-1,1)=Fload;
F_R(end-1,2)=1;
freedofs = 4:length(K_R);
U_R(freedofs,:) = K_R(freedofs,freedofs)\F_R(freedofs,:); % Solve the FEM 


%% GRADIENTS
du = zeros(nely*(nelx-2)*ncomp,1);
u=0;
el = 1;
for i=1:ncomp
    % The DOFs are calculated back to each component
    edofs = [1:ndofs] + (length(mdofs_r)-3)*(i-1);
    U_r = U_R(edofs,:); 
    U = T_r*U_r + Q0;
    U(alldofs_r,:) = U;
    U = U(4:end-3,:); %Detailed Component Stiffness Matrix 
    for elx = 1:nelx
      for ely = 1:nely/2 
        dK = fsparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1),0);
        n1 = (nely+1)*(elx-1)+ely; 
        n2 = (nely+1)* elx   +ely;
        edofs = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
        u = u + x(ely,elx+(i-1)*nelx)^(penal)*U(edofs,2)'*KE*U(edofs,1);
        if elx > 1 && elx < nelx %The first and last row of the elements remain unchanged
            du(el) = - U(edofs,2)'*penal*x(ely,elx+(i-1)*nelx)^(penal-1)*KE*U(edofs,1)/uc;   
            el = el + 1;
        end
      end
      if elx > 1 && elx < nelx
        du(el: el + nely/2-1,:) = du(linspace(el-1,el-nely/2,(nely/2)));
        el = el + nely/2;
      end 
      
    end   
    %Filter
    du(1+(i-1)*(nelx-2)*nely:(nelx-2)*i*nely)   = H*(reshape(x(:,2+(i-1)*nelx:i*nelx-1),[(nelx-2)*nely,1]).*...
                                            du(1+(i-1)*(nelx-2)*nely:(nelx-2)*i*nely))./...
                                            Hs./reshape(x(:,2+(i-1)*nelx:i*nelx-1),[(nelx-2)*nely,1]);
end 
%Since only half of the elemtents is considered due to symmetry, the other half needs to be add upp afterwards
u = 2*u; 



%% MMA OPTIMIZATION
du= reshape(du,[(nelx-2)*nely*ncomp,1]);
c = u/uc - 1;
dc = du;  
dm = ones(nely,(nelx-2)*ncomp)/(nely*nelx)*100;
dm= reshape(dm,[(nelx-2)*nely*ncomp,1]);
xval =  [reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);reshape(x(:,nelx+2:(ncomp*nelx)-1),[(nelx-2)*nely,1])]; 
f0val =m;     
df0dx= dm;  
df0dx2= 0*df0dx;
fval=c;          
dfdx = dc;
dfdx2=0*dfdx;


disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',m) ...
   ' Displ.: ' sprintf('%6.3f',u) ' change_x: ' sprintf('%6.3f',0 )])
% PLOT DENSITIES  
colormap(gray); imagesc(-x,[-1,1e-3]); axis equal; axis tight; axis off;daspect([b a 1 ]);pause(1e-6);


%% START ITERATION
conv = 0;
while conv == 0
    loop = loop + 1;
    
    [xmma,ymma,zmma,lam,xsi,eta,mu,zet,ss,low,upp] = ...
        mmasub_old(mm,nn,loop,xval,xmin,xmax,xold1,xold2, ...
        f0val,df0dx,df0dx2,fval,dfdx,dfdx2,low,upp,aa0,aa,cc,dd);
    xold2 = xold1;
    xold1 = xval;
    xval = xmma;
    
    x(:,2:nelx-1) = reshape(xval(1:(nelx-2)*nely),[nely,nelx-2]);
    x(:,nelx+2:nelx*ncomp-1) = reshape(xval((nelx-2)*nely+1:end),[nely,nelx-2]);

    ndofs = 2*(nelx+1)*(nely+1)-4*(nely+1)+6;
    K_R = sparse(ncomp*ndofs-3,ncomp*ndofs-3);
    m=0;
    for i=1:ncomp
        sK = reshape(KE(:)*(x(nelx*nely*(i-1)+1:(nelx*nely*i)).^penal),64*nelx*nely,1);
        K_ = sparse(iK,jK,sK); K_ = (K_+K_')/2;
        m = sum(x(nelx*nely*(i-1)+1:(nelx*nely*i)))/(nely*nelx)*100;
        K = fsparse(2*(nelx+1)*(nely+1)+6, 2*(nelx+1)*(nely+1)+6,0);
        K(4:end-3,4:end-3) = K_;
        edofs = [1:length(mdofs_r)] + (length(mdofs_r)-3)*(i-1);
        K_R(edofs,edofs) = K_R(edofs,edofs) + T_rt*K(alldofs_r,alldofs_r)*T_r;
    end  


    U_R = zeros(length(K_R),2); 
    freedofs = 4:length(K_R);
    U_R(freedofs,:) = K_R(freedofs,freedofs)\F_R(freedofs,:);


    %Gradients
    du = zeros(nely*(nelx-2)*ncomp,1);
    u=0;
    el = 1;
    for i=1:ncomp
        % The DOFs are calculated back to each component
        edofs = [1:ndofs] + (length(mdofs_r)-3)*(i-1);
        U_r = U_R(edofs,:); 
        U = T_r*U_r + Q0;
        U(alldofs_r,:) = U;
        U = U(4:end-3,:); %Detailed Component Stiffness Matrix 
        for elx = 1:nelx
          for ely = 1:nely/2 % A symmetry condition is set in order to remove k21, k31
            dK = fsparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1),0);
            n1 = (nely+1)*(elx-1)+ely; 
            n2 = (nely+1)* elx   +ely;
            edofs = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1; 2*n2+2; 2*n1+1; 2*n1+2];
            u = u + x(ely,elx+(i-1)*nelx)^(penal)*U(edofs,2)'*KE*U(edofs,1);
            if elx > 1 && elx < nelx %The first and last row of the elements remain unchanged
                du(el) = - U(edofs,2)'*penal*x(ely,elx+(i-1)*nelx)^(penal-1)*KE*U(edofs,1)/uc;   
                el = el + 1;
            end
          end
          if elx > 1 && elx < nelx
            du(el: el + nely/2-1,:) = du(linspace(el-1,el-nely/2,(nely/2)));
            el = el + nely/2;
          end 

        end   
        %Filter
        du(1+(i-1)*(nelx-2)*nely:(nelx-2)*i*nely)   = H*(reshape(x(:,2+(i-1)*nelx:i*nelx-1),[(nelx-2)*nely,1]).*...
                                                du(1+(i-1)*(nelx-2)*nely:(nelx-2)*i*nely))./...
                                                Hs./reshape(x(:,2+(i-1)*nelx:i*nelx-1),[(nelx-2)*nely,1]);
    end 
    %Since only half of the elemtents is considered due to symmetry, the other half needs to be add upp afterwards
    u = 2*u; 

    % MMA OPTIMIZATION
    du= reshape(du,[(nelx-2)*nely*ncomp,1]);
    c = u/uc - 1;
    dc = du;  
    dm = ones(nely,(nelx-2)*ncomp)/(nely*nelx)*100;
    dm= reshape(dm,[(nelx-2)*nely*ncomp,1]); 
    
    xval =  [reshape(x(:,2:nelx-1),[(nelx-2)*nely,1]);reshape(x(:,nelx+2:(ncomp*nelx)-1),[(nelx-2)*nely,1])];      % For the MMA-Algorithm
    f0val =m;     
    df0dx= dm;  
    df0dx2= 0*df0dx;
    fval=c;          
    dfdx = dc;
    dfdx2=0*dfdx;
    
    
    % Convergence Check
    change_x = max(abs(xval-xold1));
    feasible_f = c;
    
    if loop >= MaxIterations 
       conv =1;
       exitflag = 0;
    elseif (change_x < StepTolerance) && feasible_f < ConstraintTolerance
        conv=1;
        exitflag =2;
    end 
    
    % PRINT RESULTS
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',m) ...
   ' Displ.: ' sprintf('%6.3f',u) ' change_x: ' sprintf('%6.3f',change_x )])
    % PLOT DENSITIES  
     colormap(gray); imagesc(-x,[-1,1e-3]); axis equal; axis tight; axis off;daspect([b a 1 ]);pause(1e-6);

end 


%% In order to calculate the reference component stiffness matrices, now the normal process of Guyan Reduction + RBE2 is applied

%% Guyan
K = fsparse(2*(nelx+1)*(nely+1), 2*(nelx+1)*(nely+1),0);
K_g = fsparse(2*2*(nely+1)+6,2*2*(nely+1)+6,0);
alldofs0_g   = [1:length(K)];
mdofs_g = [1:(2*(nely+1)),length(K)-(2*(nely+1))+1:length(K)];
sdofs_g = setdiff(alldofs0_g,mdofs_g);
alldofs_g = [mdofs_g, sdofs_g];   

%% RBE2
alldofs0_r = 1:length(K_g);                   %All dofs in original order
sdofs_r = [4:length(K_g)-3];             %Dofs that are to be removed
mdofs_r = setdiff(alldofs0_r,sdofs_r);            %Dofs that remain
alldofs_r = [mdofs_r,sdofs_r];                    %New order, sdofs are at the end
newdofs_r = zeros(length(alldofs_r),1);           %Nonzeros will remove the condensed nodes
newdofs(mdofs_r) =  1:length(mdofs_r);          %Accesing the new order with the old one


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

Q = fsparse(size(C,1),1,0);                       
%Set up model for the unconstrained case
Tsm = -C(:,sdofs_r)\C(:,mdofs_r);
Ti = speye(length(mdofs_r));
T_r = [Ti;Tsm];
T_rt = transpose(T_r);
Q0 = [fsparse(length(mdofs_r),1,0);C(:,sdofs_r)\Q];


K_rg = zeros(6,6,2); 
m=zeros(1,2);
for i=1:ncomp
    sK = reshape(KE(:)*(x(nelx*nely*(i-1)+1:(nelx*nely*i)).^penal),64*nelx*nely,1);
    K_ = sparse(iK,jK,sK); K_ = (K_+K_')/2;
    m(i) = sum(x(nelx*nely*(i-1)+1:(nelx*nely*i)))/(nely*nelx)*100;
    %Guyan 
    K_g = fsparse(2*2*(nely+1)+6,2*2*(nely+1)+6,0);
    Kss = K_(sdofs_g,sdofs_g);
    Ksm = K_(sdofs_g,mdofs_g);
    T = [eye(length(mdofs_g)); -Kss\Ksm];
    K_g(4:end-3,4:end-3) = transpose(T)*K_(alldofs_g,alldofs_g)*T;
    K_rg(:,:,i) = T_rt*K_g(alldofs_r,alldofs_r)*T_r;
end  

%1
K_rg(1,1,1)
%K_rg(2,1,1)
K_rg(2,2,1)
%K_rg(3,1,1)
K_rg(3,3,1)
K_rg(6,6,1)
m(1)

%2
K_rg(1,1,2)
%K_rg(2,1,2)
K_rg(2,2,2)
%K_rg(3,1,2)
K_rg(3,3,2)
K_rg(6,6,2)
m(2)




