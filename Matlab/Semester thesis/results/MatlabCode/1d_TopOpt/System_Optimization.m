close all
clc
clear all 

rng('shuffle')

%Parameters
ConstraintTolerance = 1e-3;
uc = 1; % mm
lx= [300,300];
mload = 5e-3; % t
g = 9.806e3; % t/s^2
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
Ic = 1/12.*(Hc^3*Wc - h^3*wc);
Fload = mload*g; % vertical load F
ncomp=2;


%Bounds
lb = [];
ub = [];
for n=1:ncomp
    l = lx(n);

    %Lower Bound of Stiffness with the minimal width W
    %Bar Element
    A_lb = Hc*tw;
    KEr = A_lb*E/l*...
        [ 1 -1 
         -1  1];
     
    %Beam Element     
    k = [12 , 6*l , -12 ,  4*l^2, -6*l , 2*l^2];
    I_lb = 1/12.*(Hc^3*tw);
    KEb = I_lb*E/l^3*...
        [ k(1) k(2) k(3) k(2)
        k(2) k(4) k(5) k(6)
        k(3) k(5) k(1) k(5)
        k(2) k(6) k(5) k(4)];
    %3 DoF Stiffness Matrix
    KE = zeros(6,6);
    KE([1,4],[1,4]) = KEr;
    KE([2,3,5,6],[2,3,5,6]) = KEb;
    lb = [lb; 0.8*[KE(1,1);KE(2,2);KE(3,3);KE(6,6)]];

    %Upper Bound of Stiffness with the maximum width W 
    %Bar Element
    KEr = Ac*E/l*...
        [ 1 -1 
         -1  1];
    
    %Beam Element   
    k = [12 , 6*l , -12 ,  4*l^2, -6*l , 2*l^2];
    KEb = Ic*E/l^3*...
        [ k(1) k(2) k(3) k(2)
        k(2) k(4) k(5) k(6)
        k(3) k(5) k(1) k(5)
        k(2) k(6) k(5) k(4)];
    
    KE = zeros(6,6);
    KE([1,4],[1,4]) = KEr;
    KE([2,3,5,6],[2,3,5,6]) = KEb;
    ub = [ub; 1.05*[KE(1,1);KE(2,2);KE(3,3);KE(6,6)]];

end 


nDVs = 8; % number of variables


file = load("Figuren_1110/Ratio01_2/Mass_Estimator/ANN_r_0_97316_samples_9390_Wc_40_l_300_nel_10.mat");
ANN = file.net;

file = load("Figuren_1110/Ratio01_2/Feasibility_Estimator_TC/SVM_acc_0_88147_false_pos_0_14692_true_pos_0_91815_C_1_samples_9659_4216_Wc_40_l_300_nel_10.mat");
SVM = file.SVM;
ScoreSVM = fitSVMPosterior(SVM);

results_X = zeros(8,10);
results_y = zeros(2,10);

%% Multiple Optimization Runs due to stochastical nature of the Particle Swarm Optimization
for i=1:20
    

    costFcn = @(x)objFn(x,lx,ncomp,Fload,uc,ANN,SVM,ScoreSVM,ConstraintTolerance);
    options = optimoptions('particleswarm','SwarmSize',400,'Display','iter','UseParallel',true,'MaxIterations',800,'SelfAdjustmentWeight',1.63,'SocialAdjustmentWeight', 0.62, 'InertiaRange',[0.657,0.657],'HybridFcn','fmincon'); % 400->1600
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

    D = K(4:9,4:9)\[0;0;0;0;Fload;0]

    max(0,D(5)-uc - ConstraintTolerance)


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
        y3 = max(0,abs(D(5))-uc - ConstraintTolerance);
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



