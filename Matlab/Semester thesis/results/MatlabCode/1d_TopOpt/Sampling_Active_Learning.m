clear all 
close all
clc
addpath('fsparse')
rng(0)

%% Parameters
nel = 10; % elements
lt = 300; % mm, total length
E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3, Dichte 
mload = 5e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; % Load
nu = 0.33; % -, Poisson's Ratio
uc = 1; % mm, Max Displacement
th = 3; % mm
tw = 1; %mm
Hc = 40; % mm
Wc = Hc; % mm
h = Hc - 2*th; % mm
wc = Wc - tw; % mm, I-Stahl
Ac = Hc*Wc - h*wc; % mm^2
Ic = 1/12.*(Hc^3*Wc - h^3*wc); % moment of inertia
W_lb = ones(nel,1)*tw; % Lower design space
W_ub = ones(nel,1)*Wc; % Upper design space

%% Stiffness Lower Bounds without FEM
%Bar Element
l = lt;
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
x_lb = 0.8*[KE(1,1);KE(2,2);KE(3,3);KE(6,6)];

%% Stiffness Upper Bounds without FEM
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
x_ub = 1.05*[KE(1,1);KE(2,2);KE(3,3);KE(6,6)];

%% Feasible Seeds with FEM
%Bar Element
l = lt/nel;
KEr = E/l*...
        [ 1 -1 
         -1  1];

%Beam Element
k = [12 , 6*l , -12 ,  4*l^2, -6*l , 2*l^2];
KEb = E/l^3*...
    [ k(1) k(2) k(3) k(2)
    k(2) k(4) k(5) k(6)
    k(3) k(5) k(1) k(5)
    k(2) k(6) k(5) k(4)];


N_class=[800,1600:800:8800]; % 8800 Samples

frac = length(N_class)-1;
X = NaN(4,N_class(end));
y = NaN(1,N_class(end));
y_class = -1*ones(1,N_class(end));
X_class = zeros(4,N_class(end));


%% Random combinations of 10 finite element widths *800
% Sampling along linear cross section slope
% ratio=1, sampling the whole design space
% ratio=0.1, 0.1 deviation from linear slope

ratio = 1;
deltax = ratio*(W_ub(1) - W_lb(1));
x = W_lb(1) + (W_ub(1) -W_lb(1) - deltax)*linspace(0,1,N_class(1))'.*ones(N_class(1),10)+ lhsdesign(N_class(1),nel)*deltax;  


for i = 1:N_class(1)
    K = fsparse((nel+1)*3, (nel+1)*3,0);
    m = 0.;
    for el = 1:nel
        n1 =  3*(el-1)+1; 
        n2 =  3*el+1;
        edofr = [n1;n2];
        edofb = [n1+1;n1+2; n2+1;n2+2];
        w = x(i,el) - tw;
        A = Hc*x(i,el) - h*w; 
        m = m + A/Ac/nel*100;
        I = 1/12.*(Hc^3*x(i,el)-h^3*w);
        K(edofr,edofr) = K(edofr,edofr) + A*KEr;
        K(edofb,edofb) = K(edofb,edofb) + I*KEb;
    end
    % Guyan
    alldofs0_g     = [1:3*(nel+1)];
    mdofs_g = [1,2,3,(nel+1)*3-2,(nel+1)*3-1,(nel+1)*3];
    sdofs_g = setdiff(alldofs0_g,mdofs_g);
    alldofs_g = [mdofs_g, sdofs_g];  
    Kss = K(sdofs_g,sdofs_g);
    Ksm = K(sdofs_g,mdofs_g);
    InvKss = Kss\eye(size(Kss));
    T = [eye(length(mdofs_g)); -InvKss*Ksm];
    K_g = transpose(T)*K(alldofs_g,alldofs_g)*T;
    X_class(:,i) = [K_g(1,1);K_g(2,2),;K_g(3,3);K_g(6,6)];  
    y_class(:,i) = 1;

end 

%%
figure(100)
subplot(3,3,1)
scatter(X_class(1,:),X_class(2,:),20,y_class,'filled');xlabel('k11');ylabel('k22'); hold on;
subplot(3,3,2)
scatter(X_class(1,:),X_class(3,:),20,y_class,'filled');xlabel('k11');ylabel('k33'); hold on;
subplot(3,3,3)
scatter(X_class(1,:),X_class(4,:),20,y_class,'filled');xlabel('k11');ylabel('k66'); hold on;
subplot(3,3,4)
scatter(X_class(2,:),X_class(3,:),20,y_class,'filled');xlabel('k22');ylabel('k33'); hold on;
subplot(3,3,5)
scatter(X_class(2,:),X_class(4,:),20,y_class,'filled');xlabel('k22');ylabel('k66'); hold on;
subplot(3,3,7)
scatter(X_class(3,:),X_class(4,:),20,y_class,'filled');xlabel('k33');ylabel('k66'); hold on;

%%
SVM = fitcsvm(X_class(:,1:N_class(1))',y_class(1:N_class(1)),'KernelFunction','gaussian','KernelScale','auto',...
    'Standardize',true,'OutlierFraction',0.1);

%Sampling
N_temp_0 = 1e4;
X_temp_1 = x_lb(1) + rand(N_temp_0,1)*(x_ub(1)-x_lb(1));
X_temp_2 = x_lb(2) + rand(N_temp_0,1)*(x_ub(2)-x_lb(2));
X_temp_3 = x_lb(3) + rand(N_temp_0,1)*(x_ub(3)-x_lb(3));
X_temp_4 = x_lb(4) + rand(N_temp_0,1)*(x_ub(4)-x_lb(4));


[~,score] = predict(SVM,[X_temp_1,X_temp_2,X_temp_3,X_temp_4]);
[~,indexes] = sort(score,'descend');
X_class(:,N_class(1)+1:N_class(2)) =  [X_temp_1(indexes(1:N_class(2)-N_class(1))),X_temp_2(indexes(1:N_class(2)-N_class(1))),...
    X_temp_3(indexes(1:N_class(2)-N_class(1))),X_temp_4(indexes(1:N_class(2)-N_class(1)))]';

%%
%Cubic
figure(100)
subplot(3,3,1)
scatter(X_class(1,:),X_class(2,:),20,y_class,'filled');xlabel('k11');ylabel('k22'); hold on;
subplot(3,3,2)
scatter(X_class(1,:),X_class(3,:),20,y_class,'filled');xlabel('k11');ylabel('k33'); hold on;
subplot(3,3,3)
scatter(X_class(1,:),X_class(4,:),20,y_class,'filled');xlabel('k11');ylabel('k66'); hold on;
subplot(3,3,4)
scatter(X_class(2,:),X_class(3,:),20,y_class,'filled');xlabel('k22');ylabel('k33'); hold on;
subplot(3,3,5)
scatter(X_class(2,:),X_class(4,:),20,y_class,'filled');xlabel('k22');ylabel('k66'); hold on;
subplot(3,3,7)
scatter(X_class(3,:),X_class(4,:),20,y_class,'filled');xlabel('k33');ylabel('k66'); hold on;


%%
tic
for i=1:frac
     parfor j=N_class(i)+1:N_class(i+1)
   
        K_0= X_class(:,j)'; 
        % INITIALIZE OPTIMIZATION
        epsilon=1e-3;
        feasibleflag = 0;
        classifierflag = 0;
        exitflag = NaN;
        MaxIterations=1000; 
        ConstraintTolerance = 1e-3;
        StepTolerance = 1e-5;
        lb = W_lb; 
        ub = W_ub; 
        x = ((ub-lb)/2 + lb);
        loop = 0; 
        xold1 =  x;                     % For the MMA-Algorithm
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

        K = fsparse((nel+1)*3, (nel+1)*3,0);
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

        % Guyan
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
        for el = 1:nel
            n1 =  3*(el-1)+1; 
            n2 =  3*el+1;
            edofr = [n1;n2];
            edofb = [n1+1;n1+2; n2+1;n2+2];
            w = x(el) - tw;
            A = Hc*x(el) - h*w; 
            I = 1/12.*(Hc^3*x(el)-h^3*w);
            dK = fsparse((nel+1)*3, (nel+1)*3,0);
            dK(edofr,edofr) = Hc*KEr ;
            dK(edofb,edofb) = 1/12.*(Hc^3-h^3)*KEb;
            dm(el) =  (Hc - h)/Ac/nel*100;
            dK_g_t = transpose(T)*dK(alldofs_g,alldofs_g)*T;   
            dK_g_t = [dK_g_t(1,1),dK_g_t(2,2),dK_g_t(3,3),dK_g_t(6,6)];
            dK_g(el,:) = dK_g_t./K_0;
        end 


        %Both sides
        k = zeros(mm,1);
        k(1:mm/2) = (K_g - K_0)./K_0 - epsilon;
        k(mm/2+1:end) = (K_0 - K_g)./K_0- epsilon;

        dk = zeros(nel,mm);
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


            K = fsparse((nel+1)*3, (nel+1)*3,0);
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

            % Guyan
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
            for el = 1:nel
                n1 =  3*(el-1)+1; 
                n2 =  3*el+1;
                edofr = [n1;n2];
                edofb = [n1+1;n1+2; n2+1;n2+2];
                w = x(el) - tw;
                A = Hc*x(el) - h*w; 
                I = 1/12.*(Hc^3*x(el)-h^3*w);
                dK = fsparse((nel+1)*3, (nel+1)*3,0);
                dK(edofr,edofr) = Hc*KEr ;
                dK(edofb,edofb) = 1/12.*(Hc^3-h^3)*KEb;
                dm(el) =  (Hc - h)/Ac/nel*100; 
                dK_g_t = transpose(T)*dK(alldofs_g,alldofs_g)*T;   
                dK_g_t = [dK_g_t(1,1),dK_g_t(2,2),dK_g_t(3,3),dK_g_t(6,6)];
                dK_g(el,:) = dK_g_t./K_0;
            end 


            %Both sides
            k = zeros(mm,1);
            k(1:mm/2) = (K_g - K_0)./K_0 - epsilon;
            k(mm/2+1:end) = (K_0 - K_g)./K_0- epsilon;

            dk = zeros(nel,mm);
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

            if (feasible_f < ConstraintTolerance) && (y_class(j) ~= 1)
                y_class(j) = 1;
                X_class(:,j) = K_g;  
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
            elseif (change_x < StepTolerance) && feasible_f > ConstraintTolerance 
                conv=1;
                exitflag =-2; 

            % If the steptolerance is below the limit and the design is
            % feasible: Mass Sample
            elseif (change_x < StepTolerance) && feasible_f < ConstraintTolerance 
                conv=1;
                exitflag =2;
                y_class(j) = 1;
                X_class(:,j) = K_g; 
                X(:,j)  = K_g; 
                y(j) = m;
            end
        
        end

    end 
    
    if i==frac
        break
    end
    
    C = zeros(2,2);
    C(1,2) = 2;
    C(2,1) = 1;
    SVM = fitcsvm(X_class(:,1:N_class(i+1))',y_class(1:N_class(i+1)),'Standardize',true,'KernelFunction','gaussian','Cost',C,'OptimizeHyperparameters','auto',...
       'ClassNames',[-1,1],'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','UseParallel',true,'MaxObjectiveEvaluations',50));   
   
    gcf; close;
    gcf; close;
    
    figure(i)
    subplot(3,3,1)
    scatter(X_class(1,:),X_class(2,:),20,y_class,'filled');xlabel('k11');ylabel('k22'); hold on;
    subplot(3,3,2)
    scatter(X_class(1,:),X_class(3,:),20,y_class,'filled');xlabel('k11');ylabel('k33'); hold on;
    subplot(3,3,3)
    scatter(X_class(1,:),X_class(4,:),20,y_class,'filled');xlabel('k11');ylabel('k66'); hold on;
    subplot(3,3,4)
    scatter(X_class(2,:),X_class(3,:),20,y_class,'filled');xlabel('k22');ylabel('k33'); hold on;
    subplot(3,3,5)
    scatter(X_class(2,:),X_class(4,:),20,y_class,'filled');xlabel('k22');ylabel('k66'); hold on;
    subplot(3,3,7)
    scatter(X_class(3,:),X_class(4,:),20,y_class,'filled');xlabel('k33');ylabel('k66'); hold on;
    
    % do many Iterations to get enough feasible and infeasible samples,
    % when after 20 Iterations still enough, then the missing part of one
    % kind of samples will be filled with another kind of samples.
    min_length = 0;
    i_while = 0;
    X_temp_sum = [];
    MaxIter = 0;
    flag = 0;
    while min_length <= (N_class(i+2)-N_class(i+1))/2
        
        MaxIter = MaxIter + 1
        N_temp =  fix(i^2*(1e6-N_temp_0)/ (frac-1)^2) +  N_temp_0; % int(); 1e6 --> 1e7 || 1e8 --> x
        X_temp = zeros(4,N_temp);
        
        X_temp(1,:) = x_lb(1) + rand(N_temp,1)*(x_ub(1)-x_lb(1));
        X_temp(2,:) = x_lb(2) + rand(N_temp,1)*(x_ub(2)-x_lb(2));
        X_temp(3,:) = x_lb(3) + rand(N_temp,1)*(x_ub(3)-x_lb(3));
        X_temp(4,:) = x_lb(4) + rand(N_temp,1)*(x_ub(4)-x_lb(4));
                        
        X_temp_sum = [X_temp_sum X_temp];
        [~,distance] = predict(SVM,X_temp_sum');
        dist_Mittelwert = distance(:,1);
        
        X_temp_sum_pro = X_temp_sum(:,dist_Mittelwert > 0);
        X_temp_sum_neg = X_temp_sum(:,dist_Mittelwert < 0);
        [sortedVals_pro,indexes_pro] = sort(dist_Mittelwert(dist_Mittelwert > 0));
        X_temp_sum_pro = X_temp_sum_pro(:,indexes_pro);
        if length(X_temp_sum_pro) > (N_class(i+2)-N_class(i+1))
            X_temp_sum_pro = X_temp_sum_pro(:,1:(N_class(i+2)-N_class(i+1)));
        end
        [sortedVals_neg,indexes_neg] = sort(dist_Mittelwert(dist_Mittelwert < 0),'descend');
        X_temp_sum_neg = X_temp_sum_neg(:,indexes_neg);
        if length(X_temp_sum_neg) > (N_class(i+2)-N_class(i+1))
            X_temp_sum_neg = X_temp_sum_neg(:,1:(N_class(i+2)-N_class(i+1)));
        end
        
        min_length = min(length(indexes_pro), length(indexes_neg));
        X_temp_sum = [X_temp_sum_pro X_temp_sum_neg];
        length(X_temp_sum_pro)
        length(X_temp_sum_neg)
        i_while = i_while + 1;
        
        if MaxIter > 20
            flag = 1;
            break
        end
    end
    
    if flag == 0
        X_class(:,N_class(i+1)+1:N_class(i+2)) = [X_temp_sum_pro(:,1:(N_class(i+2)-N_class(i+1))/2) X_temp_sum_neg(:,1:(N_class(i+2)-N_class(i+1))/2)];
    else
        if length(X_temp_sum_pro) >= (N_class(i+2)-N_class(i+1))
            X_class(:,N_class(i+1)+1:N_class(i+2)) = X_temp_sum_pro(:,1:(N_class(i+2)-N_class(i+1)));
        else
            X_class(:,N_class(i+1)+1:N_class(i+2)) = X_temp_sum_neg(:,1:(N_class(i+2)-N_class(i+1)));
        end
    end

end 

%%
figure (frac)
subplot(3,3,1)
scatter(X_class(1,:),X_class(2,:),20,y_class,'filled');xlabel('k11');ylabel('k22'); hold on;
subplot(3,3,2)
scatter(X_class(1,:),X_class(3,:),20,y_class,'filled');xlabel('k11');ylabel('k33'); hold on;
subplot(3,3,3)
scatter(X_class(1,:),X_class(4,:),20,y_class,'filled');xlabel('k11');ylabel('k66'); hold on;
subplot(3,3,4)
scatter(X_class(2,:),X_class(3,:),20,y_class,'filled');xlabel('k22');ylabel('k33'); hold on;
subplot(3,3,5)
scatter(X_class(2,:),X_class(4,:),20,y_class,'filled');xlabel('k22');ylabel('k66'); hold on;
subplot(3,3,7)
scatter(X_class(3,:),X_class(4,:),20,y_class,'filled');xlabel('k33');ylabel('k66'); hold on;



%%
% Train many SVM and find the best one
X_class_SVM = X_class';

feasible = (y_class==1); 
N = length(y_class); 
N_f = sum(y_class==1);
infeasible = (y_class==-1);
X_f = X_class_SVM(logical(feasible),:)'; 
y_f= ones(length(X_f),1);
X_i = X_class_SVM(logical(infeasible),:)'; 
y_i= -1*ones(length(X_i),1);

figure(99)
subplot(3,3,1)
scatter(X_class_SVM(:,1),X_class_SVM(:,2),[],y_class,'filled'); hold on;plot(4.2984e+04,  1.7108e+03,'rx'); plot(1.6040e+04, 478.3648,'rx'); xlabel('k11'); ylabel('k22');hold on;
subplot(3,3,2)
scatter(X_class_SVM(:,1),X_class_SVM(:,3),[],y_class,'filled');hold on;plot(4.2984e+04,6.0426e+07,'rx'); plot(1.6040e+04,2.3175e+07,'rx'); xlabel('k11'); ylabel('k33');
subplot(3,3,3)
scatter(X_class_SVM(:,1),X_class_SVM(:,4),[],y_class,'filled');hold on;plot(4.2984e+04,4.2987e+07,'rx'); plot(1.6040e+04, 7.2531e+06,'rx'); xlabel('k11'); ylabel('k66');
subplot(3,3,4)
scatter(X_class_SVM(:,2),X_class_SVM(:,3),[],y_class,'filled');hold on;plot(1.7108e+03,6.0426e+07,'rx'); plot(478.3648, 2.3175e+07,'rx'); xlabel('k22'); ylabel('k33');
subplot(3,3,5)
scatter(X_class_SVM(:,2),X_class_SVM(:,4),[],y_class,'filled');hold on;plot(1.7108e+03,4.2987e+07,'rx'); plot(478.3648, 7.2531e+06,'rx');xlabel('k22'); ylabel('k66');
subplot(3,3,7)
scatter(X_class_SVM(:,3),X_class_SVM(:,4),[],y_class,'filled');hold on;plot(6.0426e+07,4.2987e+07,'rx'); plot(2.3175e+07,7.2531e+06,'rx');xlabel('k33'); ylabel('k66');


name_fp = zeros(5,5);
name_tp = zeros(5,5);
name_acc = zeros(5,5); 

for i = 1:5
    for j = 1:5
        
        %Feasible
        [idx_f_train,idx_f_test]= dividerand(length(y_f),0.8,0.2,0);
        [idx_i_train,idx_i_test]= dividerand(length(y_i),0.8,0.2,0);

        X_train = [X_f(:,idx_f_train),X_i(:,idx_i_train)]; % error?
        y_train = [y_f(idx_f_train);y_i(idx_i_train)];

        X_test = [X_f(:,idx_f_test),X_i(:,idx_i_test)];
        y_test = [y_f(idx_f_test);y_i(idx_i_test)];
        feasible_test = (y_test==1);
        infeasible_test = not(feasible_test);
        
        C = zeros(2,2);
        C(1,2) = i;
        C(2,1) = 1;

        SVM = fitcsvm(X_train',y_train,'Standardize',true,'KernelFunction','gaussian','Cost',C,'OptimizeHyperparameters','auto',...
               'ClassNames',[-1,1],'HyperparameterOptimizationOptions',struct('UseParallel',true,'MaxObjectiveEvaluations',100));     
        gcf; close;gcf; close;

        [label2,PostProbs] = predict(SVM,X_test');
        feasible_pred = (label2==1);
        infeasible_pred = not(feasible_pred);
        acc = sum(feasible_pred == feasible_test)/length(feasible_test);
        true_neg = sum(infeasible_pred & infeasible_test)/sum(infeasible_test);
        false_pos = 1 - true_neg;
        true_pos =sum(feasible_pred & feasible_test)/sum(feasible_test);
        
        name_fp(i,j) = false_pos;
        name_tp(i,j) = true_pos;
        name_acc(i,j) =  acc;
        false_pos = num2str(false_pos);
        acc = num2str(acc);
        true_pos = num2str(true_pos);

        save("Figuren_1110/Ratio1_3/Feasibility_Estimator_SAL/SVM_" + "acc_0_" + acc(3:end) +  "_false_pos_0_" + false_pos(3:end)  +  "_true_pos_0_" + true_pos(3:end) + ...
            "_C_" + num2str(C(1,2)) + "_samples_" + num2str(N) + "_" + num2str(N_f) + "_Wc_" + num2str(Wc) +  "_l_" + num2str(lt) +"_nel_" + num2str(nel),"SVM") 
    
    end 
end

[row_090, col_090] = find(name_acc>0.9);
fp_090 = zeros(1,length(row_090));
for i = 1:length(row_090)
    fp_090(i) = name_fp(row_090(i), col_090(i));
end
[~, fp_idx] = min(fp_090);
best_acc = name_acc(row_090(fp_idx),col_090(fp_idx));
best_fp = name_fp(row_090(fp_idx),col_090(fp_idx));
best_tp = name_tp(row_090(fp_idx),col_090(fp_idx));

best_acc = num2str(best_acc)
best_fp = num2str(best_fp)
best_tp = num2str(best_tp)
C_1_2 = row_090(fp_idx)

file_SVM = load("Figuren_1110/Ratio1_3/Feasibility_Estimator_SAL/SVM_" + "acc_0_" + best_acc(3:end) +  "_false_pos_0_" + best_fp(3:end)  +  "_true_pos_0_" + best_tp(3:end) + ...
            "_C_" + num2str(C_1_2) + "_samples_" + num2str(N) + "_" + num2str(N_f) + "_Wc_" + num2str(Wc) +  "_l_" + num2str(lt) +"_nel_" + num2str(nel) + ".mat");
SVM = file_SVM.SVM;

%%

X_temp = X(:,~isnan(y));
y_temp = y(~isnan(y)); 

N_mass = length(y_temp);

%% Sampling Mass 
N= 9000; 
X = NaN(4,N);
X(:,1:N_mass)= X_temp;
y = NaN(1,N);
y(:,1:N_mass)= y_temp;  
X_class_2 = zeros(4,N);
y_class_2 = -1*ones(1,N);

for j=N_mass+1:N
    %Only choose feasible designs
    f=0;
    while f==0
        X_class_2(:,j) = x_lb + rand(4,1).*(x_ub-x_lb);
        label = predict(SVM,X_class_2(:,j)');
        if label == 1
           f=1;
        end
    end
end

%%
parfor j=N_mass+1:N
       
    K_0= X_class_2(:,j)'; 
    % INITIALIZE OPTIMIZATION
    epsilon=1e-3;
    feasibleflag = 0;
    classifierflag = 0;
    exitflag = NaN;
    MaxIterations= 1000;
    ConstraintTolerance = 1e-3;
    StepTolerance = 1e-5;
    lb = W_lb; 
    ub = W_ub; 
    x = ((ub-lb)/2 + lb);
    loop = 0; 
    xold1 =  x;                     % For the MMA-Algorithm
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


    K = fsparse((nel+1)*3, (nel+1)*3,0);
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

    % Guyan
    K_g = fsparse(3*2,3*2,0);
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
    for el = 1:nel
        n1 =  3*(el-1)+1; 
        n2 =  3*el+1;
        edofr = [n1;n2];
        edofb = [n1+1;n1+2; n2+1;n2+2];
        w = x(el) - tw;
        A = Hc*x(el) - h*w; 
        I = 1/12.*(Hc^3*x(el)-h^3*w);
        dK = fsparse((nel+1)*3, (nel+1)*3,0);
        dK(edofr,edofr) = Hc*KEr ;
        dK(edofb,edofb) = 1/12.*(Hc^3-h^3)*KEb;
        dm(el) =  (Hc - h)/Ac/nel*100;
        dK_g_t = transpose(T)*dK(alldofs_g,alldofs_g)*T;   
        dK_g_t = [dK_g_t(1,1),dK_g_t(2,2),dK_g_t(3,3),dK_g_t(6,6)];
        dK_g(el,:) = dK_g_t./K_0;
    end 


    %Both sides
    k = zeros(mm,1);
    k(1:mm/2) = (K_g - K_0)./K_0 - epsilon;
    k(mm/2+1:end) = (K_0 - K_g)./K_0- epsilon;

    dk = zeros(nel,mm);
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


        K = fsparse((nel+1)*3, (nel+1)*3,0);
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

        % Guyan
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
        for el = 1:nel
            n1 =  3*(el-1)+1; 
            n2 =  3*el+1;
            edofr = [n1;n2];
            edofb = [n1+1;n1+2; n2+1;n2+2];
            w = x(el) - tw;
            A = Hc*x(el) - h*w; 
            I = 1/12.*(Hc^3*x(el)-h^3*w);
            dK = fsparse((nel+1)*3, (nel+1)*3,0);
            dK(edofr,edofr) = Hc*KEr ;
            dK(edofb,edofb) = 1/12.*(Hc^3-h^3)*KEb;
            dm(el) =  (Hc - h)/Ac/nel*100;
            dK_g_t = transpose(T)*dK(alldofs_g,alldofs_g)*T;   
            dK_g_t = [dK_g_t(1,1),dK_g_t(2,2),dK_g_t(3,3),dK_g_t(6,6)];
            dK_g(el,:) = dK_g_t./K_0;
        end 


        %Both sides
        k = zeros(mm,1);
        k(1:mm/2) = (K_g - K_0)./K_0 - epsilon;
        k(mm/2+1:end) = (K_0 - K_g)./K_0- epsilon;

        dk = zeros(nel,mm);
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

        if (feasible_f < ConstraintTolerance) && (y_class_2(j) ~= 1)
            y_class_2(j) = 1;
            X_class_2(:,j) = K_g;  
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
        elseif (change_x < StepTolerance) && feasible_f > ConstraintTolerance
            conv=1;
            exitflag =-2; 

        % If the steptolerance is below the limit and the design is
        % feasible: Mass Sample
        elseif (change_x < StepTolerance) && feasible_f < ConstraintTolerance 
            conv=1;
            exitflag =2;
            y_class_2(j) = 1;
            X_class_2(:,j) = K_g; 
            X(:,j)  = K_g; 
            y(j) = m;
        end
    end

end 
% toc     

%% Plotting
figure(1000)
subplot(3,3,1)
scatter(X(1,:),X(2,:),20,y,'filled');xlabel('k11');ylabel('k22'); hold on;
subplot(3,3,2)
scatter(X(1,:),X(3,:),20,y,'filled');xlabel('k11');ylabel('k33'); hold on;
subplot(3,3,3)
scatter(X(1,:),X(4,:),20,y,'filled');xlabel('k11');ylabel('k66'); hold on;
subplot(3,3,4)
scatter(X(2,:),X(3,:),20,y,'filled');xlabel('k22');ylabel('k33'); hold on;
subplot(3,3,5)
scatter(X(2,:),X(4,:),20,y,'filled');xlabel('k22');ylabel('k66'); hold on;
subplot(3,3,7)
scatter(X(3,:),X(4,:),20,y,'filled');xlabel('k33');ylabel('k66'); hold on;

%% Creating Samples
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

%NEU
[idx2,distance2]= knnsearch(X',X_2','Distance','seuclidean');
y_2 = 1.2*y(idx2);
y_2(distance2 ==0) = y(distance2 ==0);
%NEU

X_3 =  X_class_2(:,y_class_2(N_mass+1:end) == -1);

%NEU
[idx3,distance3]= knnsearch(X',X_3','Distance','seuclidean');
y_3 = 1.2*y(idx3);
 
%NEU

X_3 = [X,X_2,X_3];
y_3 = [y,y_2,y_3];

samples = [X_3;y_3];
save("Figuren_1110\Samples_Mass\samples_" + num2str(length(y(~isnan(y)))) + "_Wc_" + num2str(Wc) +  "_l_" + num2str(lt) +"_nel_" + num2str(nel),'samples')

%%
%Classifier Samples 
% 1) Add false positive estimates from second loop to classifier
y_class_2 = y_class_2(N_mass+1:end);
y_class = [y_class, y_class_2(y_class_2 == -1)];
X_class_2 = X_class_2(:,N_mass+1:end);
X_class = [X_class,X_class_2(:,y_class_2 == -1)];

samples_class = [X_class;y_class];
save("Figuren_1110\Samples_Classification\samples_" + num2str(length(y_class)) + "_Wc_" + num2str(Wc) +  "_l_" + num2str(lt) +"_nel_" + num2str(nel),'samples_class');



