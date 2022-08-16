close all
clc
clear all
addpath('fsparse')


% Optimization Problem to solve 
% min   m
% s.t.  -epsilon < k11 - k11_0 < epsilon
%       -epsilon < k12 - k12_0 < epsilon
%       -epsilon < k13 - k13_0 < epsilon
% Design variables are the width of the beam

%Parameters
nel = 10; 
lt = 300; % mm
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

x = W_lb(1) + (W_ub(1)-W_lb(1))*rand(nel,1);
l = lt/nel;

%% Exemplary Stiffness for a Beam
% KEr = E/l*...
%         [ 1 -1 
%          -1  1];
% k = [12 , 6*l , -12 ,  4*l^2, -6*l , 2*l^2];
% KEb = E/l^3*...
%     [ k(1) k(2) k(3) k(2)
%     k(2) k(4) k(5) k(6)
%     k(3) k(5) k(1) k(5)
%     k(2) k(6) k(5) k(4)];
% K = zeros((nel+1)*3, (nel+1)*3);
% F = zeros((nel+1)*3,3);
% U = zeros((nel+1)*3,3);
% W = zeros(1,nel);
% m_0 = 0.;
% for el = 1:nel
%     n1 =  3*(el-1)+1; 
%     n2 =  3*el+1;
%     edofr = [n1;n2];
%     edofb = [n1+1;n1+2; n2+1;n2+2];
%     w = x(el) - tw;
%     A = Hc*x(el) - h*w; 
%     m_0 = m_0 + A/Ac/nel*100;
%     I = 1/12.*(Hc^3*x(el)-h^3*w);
%     K(edofr,edofr) = K(edofr,edofr) + A*KEr;
%     K(edofb,edofb) = K(edofb,edofb) + I*KEb;
% end
% alldofs0_g     = [1:3*(nel+1)];
% mdofs_g = [1,2,3,(nel+1)*3-2,(nel+1)*3-1,(nel+1)*3];
% sdofs_g = setdiff(alldofs0_g,mdofs_g);
% alldofs_g = [mdofs_g, sdofs_g];  
% Kss = K(sdofs_g,sdofs_g);
% Ksm = K(sdofs_g,mdofs_g);
% InvKss = Kss\eye(size(Kss));
% T = [eye(length(mdofs_g)); -InvKss*Ksm];
% K_0 = transpose(T)*K(alldofs_g,alldofs_g)*T;
% K_0 = [K_0(1,1),K_0(2,2),K_0(4,4),K_0(6,6)];

% Test Ergebnisse
% file = load('Figuren_1110/Ratio1_2/Parameter_results_X');
% K_0 = file.results_X(1:4,3)';
% file2 = load('Figuren_1110/Ratio1_2/Parameter_results_y');
% m_0 = file2.results_y(1,3);

% Optimale Ergebnisse für Komponent1
% K_0 = [4.2984*1e4 1.7108*1e3 6.0426*1e7 4.2987*1e7];
% m_0 = 69.2001;

% Optimale Ergebnisse für Komponent2
K_0 = [1.6040*1e4 478.3648 2.3175*1e7 7.2531*1e6];
m_0 = 29.4164;

%%
figure(1)
subplot(1,2,1);
stairs(0:l:(nel)*l,[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); hold on 
stairs(0:l:(nel)*l,-[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); 
axis([0,lt,-Wc/2,Wc/2]); xlabel('length l (mm)'); ylabel('width W (mm)');






%% Optimization
 
% INITIALIZE OPTIMIZATION
exitflag = NaN;
feasibleflag = 0;
epsilon = 1e-3;
MaxIterations= 1000;
ConstraintTolerance = 1e-3;
StepTolerance = 1e-5;
lb = W_lb; 
ub = W_ub; 
x = ((ub-lb)/2 + lb);
loop = 0; 
xold1 =  x;      % For the MMA-Algorithm
xold2 =  x; 
mm = 4*2;                                  % Number of constraints
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
K = zeros((nel+1)*3, (nel+1)*3);
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
    m = m + A/Ac/nel*100;
    I = 1/12.*(Hc^3*x(el)-h^3*w);
    K(edofr,edofr) = K(edofr,edofr) + A*KEr;
    K(edofb,edofb) = K(edofb,edofb) + I*KEb;
end
alldofs0_g     = [1:3*(nel+1)];
mdofs_g = [1,2,3,(nel+1)*3-2,(nel+1)*3-1,(nel+1)*3];
sdofs_g = setdiff(alldofs0_g,mdofs_g);
alldofs_g = [mdofs_g, sdofs_g];  
Kss = K(sdofs_g,sdofs_g);
Ksm = K(sdofs_g,mdofs_g);
InvKss = Kss\eye(size(Kss));
T = [eye(length(mdofs_g)); -InvKss*Ksm];
K_g = transpose(T)*K(alldofs_g,alldofs_g)*T;
K_g = [K_g(1,1),K_g(2,2),K_g(3,3),K_g(6,6)];


% GRADIENTS
dK_g = zeros(nel,mm/2);
dm = zeros(nel,1); 
dk = zeros(nel,mm);
for el = 1:nel
    dK = fsparse(3*(nel+1), 3*(nel+1),0);
    n1 =  3*(el-1)+1; 
    n2 =  3*el+1;
    edofr = [n1;n2];
    edofb = [n1+1;n1+2; n2+1;n2+2]; 
    dK(edofr,edofr) = Hc*KEr;
    dK(edofb,edofb) =  1/12.*(Hc^3-h^3)*KEb;
    dK_g_t = transpose(T)*dK(alldofs_g,alldofs_g)*T;   
    dK_g_t = [dK_g_t(1,1),dK_g_t(2,2),dK_g_t(4,4),dK_g_t(6,6)];
    dK_g(el,:) =  dK_g_t./K_0;
    dm(el) =  (Hc - h)/Ac/nel*100;
end

%Both sides
k = zeros(mm,1);
k(1:mm/2,1) = (K_g - K_0)./K_0 - epsilon;
k(mm/2+1:end,1) = (K_0 - K_g)./K_0 - epsilon;
dk(:,1:mm/2) = dK_g;
dk(:,mm/2+1:end) = -dK_g;

% MMA OPTIMIZATION
xval =  x;
f0val =m;     
df0dx= dm; 
df0dx2= 0*df0dx;
fval=k;          
dfdx=dk';
dfdx2=0*dfdx; 



subplot(1,2,2);
stairs(0:l:(nel)*l,[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); hold on 
stairs(0:l:(nel)*l,-[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); 
axis([0,lt,-Wc/2,Wc/2]); xlabel('length l (mm)'); ylabel('width W (mm)');


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

    K = zeros((nel+1)*3, (nel+1)*3);
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
        m = m + A/Ac/nel*100;
        I = 1/12.*(Hc^3*x(el)-h^3*w);
        K(edofr,edofr) = K(edofr,edofr) + A*KEr;
        K(edofb,edofb) = K(edofb,edofb) + I*KEb;
    end

    alldofs0_g     = [1:3*(nel+1)];
    mdofs_g = [1,2,3,(nel+1)*3-2,(nel+1)*3-1,(nel+1)*3];
    sdofs_g = setdiff(alldofs0_g,mdofs_g);
    alldofs_g = [mdofs_g, sdofs_g];  
    Kss = K(sdofs_g,sdofs_g);
    Ksm = K(sdofs_g,mdofs_g);
    InvKss = Kss\eye(size(Kss));
    T = [eye(length(mdofs_g)); -InvKss*Ksm];
    K_g = transpose(T)*K(alldofs_g,alldofs_g)*T;
    K_g = [K_g(1,1),K_g(2,2),K_g(3,3),K_g(6,6)];


    % GRADIENTS
    dK_g = zeros(nel,mm/2);
    dm = zeros(nel,1); 
    dk = zeros(nel,mm);
    for el = 1:nel
        dK = fsparse(3*(nel+1), 3*(nel+1),0);
        n1 =  3*(el-1)+1; 
        n2 =  3*el+1;
        edofr = [n1;n2];
        edofb = [n1+1;n1+2; n2+1;n2+2]; 
        dK(edofr,edofr) = Hc*KEr;
        dK(edofb,edofb) =  1/12.*(Hc^3-h^3)*KEb;
        dK_g_t = transpose(T)*dK(alldofs_g,alldofs_g)*T;   
        dK_g_t = [dK_g_t(1,1),dK_g_t(2,2),dK_g_t(4,4),dK_g_t(6,6)];
        dK_g(el,:) =  dK_g_t./K_0;
        dm(el) =  (Hc - h)/Ac/nel*100;
    end

    %Both sides
    k = zeros(mm,1);
    k(1:mm/2,1) = (K_g - K_0)./K_0 - epsilon;
    k(mm/2+1:end,1) = (K_0 - K_g)./K_0 - epsilon;

    dk(:,1:mm/2) = dK_g;
    dk(:,mm/2+1:end) = -dK_g;

    % MMA OPTIMIZATION
    xval =  x;
    f0val =m;     
    df0dx= dm; 
    df0dx2= 0*df0dx;
    fval=k;          
    dfdx=dk';
    dfdx2=0*dfdx; 

    % Convergence Check
    change_x = max(abs(xval-xold1)./xold1);
    feasible_f = max(k);


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
    elseif (change_x < StepTolerance) && feasible_f > ConstraintTolerance
        conv=1;
        exitflag =-2; 
    % If the steptolerance is below the limit and the design is
    % feasible: Mass Sample
    elseif (change_x < StepTolerance) && feasible_f < ConstraintTolerance 
        conv=1;
        exitflag =2;
    end
    
    figure (1)
    subplot(1,2,2); hold off 
    stairs(0:l:(nel)*l,[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); hold on 
    stairs(0:l:(nel)*l,-[x;x(end)]/2,'color',[10 10 10]/40,'Linewidth',1.5); 
    axis([0,lt,-Wc/2,Wc/2]); xlabel('length l (mm)'); ylabel('width W (mm)');
    
    disp([' It.: ' sprintf('%4i',loop) ' Obj.: ' sprintf('%10.4f',m) ...
        ' Stiffness: ' sprintf('%6.3f',max(k)) ' change_x: ' sprintf('%6.3f',change_x )])

end 


K_0
m_0

K_g
m


    