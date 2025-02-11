close all
clc
clear all 


%Parameters
E= 70e3; % MPa
rho = 2.7e-9; % t/mm^3 
mload = 60e-3; % t
g = 9.806e3; % t/s^2
Fload = mload*g; %Load
nu = 0.33; % -
penal = 4;
uc = 1; % mm

%Number of elements
lx=300;
ly=75;
lz=25;
nelx = 30; %60
nely = 10; %10 
nelz = 4;
ndofs = 3;
a = 0.49*lx/nelx; %x
b = 0.49*ly/nely; %y
c = 0.49*lz/nelz; %z


%% Binary Classifier
file = load("Versuch_7500\Samples_Classification\samples_7530_lx_300_ly_75_lz_25_nelx_30_nely_10_nelz_4.mat");

X = file.samples_class(1:4,:)';
y = file.samples_class(5,:)';
feasible = (y==1); 
N = length(y); 
N_f = sum(y==1);
infeasible = (y==-1);
X_f = X(logical(feasible),:)'; 
y_f= ones(length(X_f),1);
X_i = X(logical(infeasible),:)'; 
y_i= -1*ones(length(X_i),1);

figure(1)
subplot(3,3,1)
scatter(X(:,1),X(:,2),[],y,'filled'); 
subplot(3,3,2)
scatter(X(:,1),X(:,3),[],y,'filled');
subplot(3,3,3)
scatter(X(:,1),X(:,4),[],y,'filled');
subplot(3,3,4)
scatter(X(:,2),X(:,3),[],y,'filled');
subplot(3,3,5)
scatter(X(:,2),X(:,4),[],y,'filled');
subplot(3,3,7)
scatter(X(:,3),X(:,4),[],y,'filled');

name_fp = strings(8,1);
name_tp = strings(8,1);
name_acc = strings(8,1); 

list_fp = zeros(60,1);
list_tp = zeros(60,1);
list_acc = zeros(60,1); 

C = zeros(2,2);
C(1,2) = 4;
C(2,1) = 1;

iter = 1; loop = 1; count =1;
while iter <  8

    %Train Classifier
    %Feasible
    [idx_f_train,idx_f_test]= dividerand(length(y_f),0.8,0.2,0);
    [idx_i_train,idx_i_test]= dividerand(length(y_i),0.8,0.2,0);

    X_train = [X_f(:,idx_f_train,:),X_i(:,idx_i_train,:)];
    y_train = [y_f(idx_f_train);y_i(idx_i_train)];

    X_test = [X_f(:,idx_f_test),X_i(:,idx_i_test)];
    y_test = [y_f(idx_f_test);y_i(idx_i_test)];
    feasible_test = (y_test==1);
    infeasible_test = not(feasible_test);

    %ClassificationCosts
    if loop > 5
        loop = 1;
        C(1,2) = C(1,2) + 1;
    end 


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
    
    list_fp(count,1) = false_pos;
    list_tp(count,1) = true_pos;
    list_acc(count,1) =  acc;


    if false_pos < 0.05 && ...
            (~any(strcmp(name_fp,num2str(false_pos))) || ~any(strcmp(name_tp,num2str(true_pos))) || ~any(strcmp(name_acc,num2str(acc)))) ...  
           && true_pos > 0.85
       
        false_pos = num2str(false_pos);
        acc = num2str(acc);
        true_pos = num2str(true_pos);
        name_fp(iter,1) = false_pos;
        name_tp(iter,1) = true_pos;
        name_acc(iter,1) =  acc;
        save("Versuch_7500/Feasibility_Estimator/SVM_" + ...
            "acc_0_" + acc(3:end) +  "_false_pos_0_" + false_pos(3:end)  +  "_true_pos_0_" + true_pos(3:end) + ...
            "_C_" + num2str(C(1,2)) + "_samples_" + num2str(N) + "_" + num2str(N_f) + ...
            "_lx_" + num2str(lx) + "_ly_" + num2str(ly) + "_lz_" + num2str(lz) + ...
            "_nelx_" + num2str(nelx) + "_nely_" + num2str(nely) + "_nelz_" + num2str(nelz),"SVM");
        iter = iter+1;
    end 
    
    count = count +1;
    loop = loop + 1;

    if count == 60
        break
    end

end