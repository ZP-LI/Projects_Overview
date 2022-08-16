clear all
close all
clc


% INITIALIZE OPTIMIZATION
ncomp = 2;
nelcomp = 10;
ltcomp = 300; % mm
nel = nelcomp*ncomp;
lt = ltcomp*ncomp;
E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3 
nu = 0.33; % -
th = 3; % mm
tw = 1; %mm
Hc = 40; % mm
Wc = Hc; %mm
h = Hc - 2*th;%%
wc = Wc - tw; % mm
Ac = Hc*Wc - h*wc; % mm
Ic = 1/12.*(Hc^3*Wc - h^3*wc); % moment of inertia for a i-beam
W_lb = ones(nel,1)*tw; % Lower bound for the beam width
W_ub = ones(nel,1)*Wc; % Upper bound for the bevam width
mload = 5e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; %Load
uc = 1; % mm



% INITIALIZE OPTIMIZATION
MaxIterations= 500;
epsilon = 1e-3;
ConstraintTolerance = 1e-4;
StepTolerance = 1e-8;
lb(1:nel,1) = W_lb; 
ub(1:nel,1) = W_ub; 
x = ((ub-lb)/2 + lb);
loop = 0; 
xold1 =  x;      % For the MMA-Algorithm
xold2 =  x; 
mm = 1;                                  % Number of constraints
nn=nel;                           % Number of designvariables
aa0=1;                   
aa=zeros(mm,1);
cc=1e3*ones(mm,1);
dd=zeros(mm,1);
xmin = lb;         % Lower bounds of design variables
low = xmin;
xmax = ub;               % Upper bounds of design variables
upp = xmax;


%INITIAL ANALYSIS
l = lt/nel;
KEr = E/l*...
        [ 1 -1 
         -1  1];

k = [12 , 6*l , -12 ,  4*l^2, -6*l , 2*l^2];
KEb = E/l^3*...
    [ k(1) k(2) k(3) k(2)
    k(2) k(4) k(5) k(6)
    k(3) k(5) k(1) k(5)
    k(2) k(6) k(5) k(4)];

K = zeros((nel+1)*3,(nel+1)*3);
F = zeros((nel+1)*3,3);
U = zeros((nel+1)*3,3);
W = zeros(1,nel);
m = 0.;
for el = 1:nel
    n1 =  3*(el-1)+1; 
    n2 =  3*el+1;
    edofr = [n1;n2];
    edofb = [n1+1;n1+2; n2+1;n2+2];
    w = x(el) - tw;
    A = Hc*x(el) - h*w; 
    m = m + A/Ac/nelcomp*100;
    I = 1/12.*(Hc^3*x(el)-h^3*w);
    K(edofr,edofr) = K(edofr,edofr) + A*KEr;
    K(edofb,edofb) = K(edofb,edofb) + I*KEb;
end

% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
F(3*(nel+1)-1,1) = Fload;
F(3*(nel+1)-1,2) = 1;
fixeddofs   = [1,2,3];
alldofs     = [1:3*(nel+1)];
freedofs    = setdiff(alldofs,fixeddofs);
% SOLVING
U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);      
U(fixeddofs,:)= 0;

u = 0.;
du = zeros(nel,1);
dm = zeros(nel,1);
for el = 1:nel
    n1 =  3*(el-1)+1; 
    n2 =  3*el+1;
    edof = [n1;n1+1;n1+2; n2;n2+1;n2+2];
    w = x(el) - tw;
    A = Hc*x(el) - h*w; 
    I = 1/12.*(Hc^3*x(el)-h^3*w);
    Ue = U(edof,1);
    Ue2 = U(edof,2);
    
    KE = zeros(6,6);
    dKE = zeros(6,6);
    KE([1,4],[1,4]) = A*KEr;
    KE([2,3,5,6],[2,3,5,6]) = I*KEb;
    dKE([1,4],[1,4]) = -(Hc-h)*KEr;
    dKE([2,3,5,6],[2,3,5,6]) = -1/12.*(Hc^3-h^3)*KEb;
    
    u = u + Ue2'*KE*Ue;
    du(el) = Ue2'*dKE*Ue/uc;
    dm(el) =  (Hc - h)/Ac/nelcomp*100;
end

c(1) = u/uc-1 - epsilon;
dc(:,1) = du;

% MMA OPTIMIZATION
xval =  x;
f0val =m;     
df0dx= dm; 
df0dx2= 0*df0dx;
fval=c;          
dfdx=dc';
dfdx2=0*dfdx; 


%Plot
figure(1)
stairs(0:l:(nel)*l,[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); hold on 
stairs(0:l:(nel)*l,-[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); 
axis([0,lt,-Wc/2,Wc/2]); xlabel('length l (mm)'); ylabel('width W (mm)');

disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',m) ...
   ' Stiffness: ' sprintf('%6.3f',max(c)) ' change_x: ' sprintf('%6.3f',0 )])

% START ITERATION
conv = 0;
while conv == 0
    loop = loop + 1;

    % MMA OPTIMIZATION
    [xmma,ymma,zmma,lam,xsi,eta,mu,zet,ss,low,upp] = ...
        mmasub(mm,nn,loop,xval,xmin,xmax,xold1,xold2, ...
        f0val,df0dx,df0dx2,fval,dfdx,dfdx2,low,upp,aa0,aa,cc,dd);

    f0valold = f0val;    
    xold2 = xold1;
    xold1 = xval;
    xval = xmma;
    x = xval;
    
    %INITIAL ANALYSIS
    
    l = lt/nel;
    KEr = E/l*...
        [ 1 -1 
         -1  1];
     
    k = [12 , 6*l , -12 ,  4*l^2, -6*l , 2*l^2];
    KEb = E/l^3*...
        [ k(1) k(2) k(3) k(2)
        k(2) k(4) k(5) k(6)
        k(3) k(5) k(1) k(5)
        k(2) k(6) k(5) k(4)];

    K = zeros((nel+1)*3,(nel+1)*3);
    F = zeros((nel+1)*3,3);
    U = zeros((nel+1)*3,3);
    W = zeros(1,nel);
    m = 0.;
    for el = 1:nel
        n1 =  3*(el-1)+1; 
        n2 =  3*el+1;
        edofr = [n1;n2];
        edofb = [n1+1;n1+2; n2+1;n2+2];
        w = x(el) - tw;
        A = Hc*x(el) - h*w; 
        m = m + A/Ac/nelcomp*100;
        I = 1/12.*(Hc^3*x(el)-h^3*w);
        K(edofr,edofr) = K(edofr,edofr) + A*KEr;
        K(edofb,edofb) = K(edofb,edofb) + I*KEb;
    end
    
    % DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
    F(3*(nel+1)-1,1) = Fload;
    F(3*(nel+1)-1,2) = 1;
    fixeddofs   = [1,2,3];
    alldofs     = [1:3*(nel+1)];
    freedofs    = setdiff(alldofs,fixeddofs);
    % SOLVING
    U(freedofs,:) = K(freedofs,freedofs) \ F(freedofs,:);      
    U(fixeddofs,:)= 0;

    u = 0.;
    du = zeros(nel,1);
    dm = zeros(nel,1);
    for el = 1:nel
        n1 =  3*(el-1)+1; 
        n2 =  3*el+1;
        edof = [n1;n1+1;n1+2; n2;n2+1;n2+2];
        w = x(el) - tw;
        A = Hc*x(el) - h*w; 
        I = 1/12.*(Hc^3*x(el)-h^3*w);
        Ue = U(edof,1);
        Ue2 = U(edof,2);
    
        KE = zeros(6,6);
        dKE = zeros(6,6);
        KE([1,4],[1,4]) = A*KEr;
        KE([2,3,5,6],[2,3,5,6]) = I*KEb;
        dKE([1,4],[1,4]) = -(Hc-h)*KEr;
        dKE([2,3,5,6],[2,3,5,6]) = -1/12.*(Hc^3-h^3)*KEb;
    
        u = u + Ue2'*KE*Ue;
        du(el) = Ue2'*dKE*Ue/uc;
        dm(el) =  (Hc - h)/Ac/nelcomp*100;
    end

    c(1) = u/uc-1 - epsilon;
    dc(:,1) = du;

    % MMA OPTIMIZATION
    xval =  x;
    f0val =m;     
    df0dx= dm; 
    df0dx2= 0*df0dx;
    fval=c;          
    dfdx=dc';
    dfdx2=0*dfdx; 

    % Convergence Check
    change_x = max(max(abs(xval-xold1)/xold1));
    feasible_f = max(c);

    if loop >= MaxIterations && feasible_f > ConstraintTolerance
       conv =1;
       exitflag = -2;
    elseif loop >= MaxIterations && feasible_f < ConstraintTolerance
       conv =1;
       exitflag = 0;
    elseif (change_x < StepTolerance) && feasible_f < ConstraintTolerance
        conv=1;
        exitflag =2;
    end
    
    
    % PRINT RESULTS
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',m) ...
   ' Stiffness: ' sprintf('%6.3f',max(c)) ' change_x: ' sprintf('%6.3f',change_x )])
    %Plot
    figure(2)
    stairs(0:l:(nel)*l,[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); hold on 
    stairs(0:l:(nel)*l,-[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); 
    axis([0,lt,-Wc/2,Wc/2]); xlabel('length l (mm)'); ylabel('width W (mm)');
    hold off 


end 

m

%Result 
%Component 1 
l = ltcomp/nelcomp;
KEr = E/l*...
        [ 1 -1 
         -1  1];
     
k = [12 , 6*l , -12 ,  4*l^2, -6*l , 2*l^2];
KEb = E/l^3*...
    [ k(1) k(2) k(3) k(2)
    k(2) k(4) k(5) k(6)
    k(3) k(5) k(1) k(5)
    k(2) k(6) k(5) k(4)];

K = zeros((nelcomp+1)*3, (nelcomp+1)*3);
m1 = 0.;
for el = 1:nelcomp
    n1 =  3*(el-1)+1; 
    n2 =  3*el+1;
    edofr = [n1;n2];
    edofb = [n1+1;n1+2; n2+1;n2+2];
    w = x(el) - tw;
    A = Hc*x(el) - h*w; 
    m1 = m1 + A/Ac/nelcomp*100;
    I = 1/12.*(Hc^3*x(el)-h^3*w);
    K(edofr,edofr) = K(edofr,edofr) + A*KEr;
    K(edofb,edofb) = K(edofb,edofb) + I*KEb;
end

alldofs0_g     = [1:3*(nelcomp+1)];
mdofs_g = [1,2,3,(nelcomp+1)*3-2,(nelcomp+1)*3-1,(nelcomp+1)*3];
sdofs_g = setdiff(alldofs0_g,mdofs_g);
alldofs_g = [mdofs_g, sdofs_g];  
Kss = K(sdofs_g,sdofs_g);
Ksm = K(sdofs_g,mdofs_g);
InvKss = Kss\eye(size(Kss));
T = [eye(length(mdofs_g)); -InvKss*Ksm];
K_g1 = transpose(T)*K(alldofs_g,alldofs_g)*T;
kappa_1 = [K_g1(1,1);K_g1(2,2),;K_g1(3,3);K_g1(6,6)]  

%Component 2 
K = zeros((nelcomp+1)*3, (nelcomp+1)*3);
m2 = 0.;
for el = 1:nelcomp
    n1 =  3*(el-1)+1; 
    n2 =  3*el+1;
    edofr = [n1;n2];
    edofb = [n1+1;n1+2; n2+1;n2+2];
    w = x(nelcomp + el) - tw;
    A = Hc*x(nelcomp + el) - h*w; 
    m2 = m2 + A/Ac/nelcomp*100;
    I = 1/12.*(Hc^3*x(nelcomp + el)-h^3*w);
    K(edofr,edofr) = K(edofr,edofr) + A*KEr;
    K(edofb,edofb) = K(edofb,edofb) + I*KEb;
end

alldofs0_g     = [1:3*(nelcomp+1)];
mdofs_g = [1,2,3,(nelcomp+1)*3-2,(nelcomp+1)*3-1,(nelcomp+1)*3];
sdofs_g = setdiff(alldofs0_g,mdofs_g);
alldofs_g = [mdofs_g, sdofs_g];  
Kss = K(sdofs_g,sdofs_g);
Ksm = K(sdofs_g,mdofs_g);
InvKss = Kss\eye(size(Kss));
T = [eye(length(mdofs_g)); -InvKss*Ksm];
K_g2 = transpose(T)*K(alldofs_g,alldofs_g)*T;
kappa_2 = [K_g2(1,1);K_g2(2,2),;K_g2(3,3);K_g2(6,6)]  

Gesamt_Masse = (m1 + m2)

optimal_samples_class = [kappa_1 kappa_2;1 1];
optimal_samples_mass = [kappa_1 kappa_2;m1 m2];

save('Samples_Classification\optimal_samples_class','optimal_samples_class')
save('Samples_Mass\optimal_samples_mass','optimal_samples_mass')

% Gesamte Masse(98.6165kg) gleich geblieben wie es in 2Dof.
% kappa_repräsentation in 2Dof: 1:[0,0002 6,0426 4,2987]*1e7
%                               2:[0,0000 2,3175 0,7253]*1e7
% kappa_repräsentation in 3Dof: 1:[0,0043 0,0002 6,0426 4,2987]*1e7
%                               2:[0,0016 0,0000 2,3175 0,7253]*1e7
% kappa_repräsentation in 3Dof in Decompose Verfarhen: 1:[0.0038 0.0002 5.2922 4.1302]*1e7
%                                                      2:[0.0028 0.0001 4.0509 3.7439]*1e7
