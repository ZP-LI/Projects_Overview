close all
clc
clear all 

%Number of elements
nel = 10;
lt = 300; % mm
Wc = 40; % mm


%% Binary Classifier
file = load("Figuren_1103/Samples_Classification/samples_9994_Wc_40_l_300_nel_10.mat");

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
scatter(X(:,1),X(:,2),[],y,'filled'); hold on;plot(4.2984e+04,  1.7108e+03,'rx'); plot(1.6040e+04, 478.3648,'rx'); xlabel('k11'); ylabel('k22');hold on;
subplot(3,3,2)
scatter(X(:,1),X(:,3),[],y,'filled');hold on;plot(4.2984e+04,6.0426e+07,'rx'); plot(1.6040e+04,2.3175e+07,'rx'); xlabel('k11'); ylabel('k33');
subplot(3,3,3)
scatter(X(:,1),X(:,4),[],y,'filled');hold on;plot(4.2984e+04,4.2987e+07,'rx'); plot(1.6040e+04, 7.2531e+06,'rx'); xlabel('k11'); ylabel('k66');
subplot(3,3,4)
scatter(X(:,2),X(:,3),[],y,'filled');hold on;plot(1.7108e+03,6.0426e+07,'rx'); plot(478.3648, 2.3175e+07,'rx'); xlabel('k22'); ylabel('k33');
subplot(3,3,5)
scatter(X(:,2),X(:,4),[],y,'filled');hold on;plot(1.7108e+03,4.2987e+07,'rx'); plot(478.3648, 7.2531e+06,'rx');xlabel('k22'); ylabel('k66');
subplot(3,3,7)
scatter(X(:,3),X(:,4),[],y,'filled');hold on;plot(6.0426e+07,4.2987e+07,'rx'); plot(2.3175e+07,7.2531e+06,'rx');xlabel('k33'); ylabel('k66');

name_fp = strings(8,1);
name_tp = strings(8,1);
name_acc = strings(8,1); 

C = zeros(2,2);
C(1,2) = 2;
C(2,1) = 1;

iter = 0; loop = 0; count = 0;
while iter <  8
    count = count +1;
    loop = loop + 1;
    %Train Classifier

    %Feasible
    [idx_f_train,idx_f_test]= dividerand(length(y_f),0.8,0.2,0);
    [idx_i_train,idx_i_test]= dividerand(length(y_i),0.8,0.2,0);

    X_train = [X_f(:,idx_f_train),X_i(:,idx_i_train)]; % error?
    y_train = [y_f(idx_f_train);y_i(idx_i_train)];

    X_test = [X_f(:,idx_f_test),X_i(:,idx_i_test)];
    y_test = [y_f(idx_f_test);y_i(idx_i_test)];
    feasible_test = (y_test==1);
    infeasible_test = not(feasible_test);

    %ClassificationCosts
    if loop > 5
        loop = 0;
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

    list_acc(count) = acc;
    list_fp(count) = false_pos;
    list_tp(count) = true_pos;

    if false_pos < 0.05 && ...
            (~any(strcmp(name_fp,num2str(false_pos))) || ~any(strcmp(name_tp,num2str(true_pos))) || ~any(strcmp(name_acc,num2str(acc)))) ...  
           && true_pos > 0.85
         
        iter = iter+1;
 
        false_pos = num2str(false_pos);
        acc = num2str(acc);
        true_pos = num2str(true_pos);
        name_fp(iter,1) = false_pos;
        name_tp(iter,1) = true_pos;
        name_acc(iter,1) =  acc;
        save("Figuren_1103/Ratio0.1_1/Feasibility_Estimator_T_C/SVM_" + "acc_0_" + acc(3:end) +  "_false_pos_0_" + false_pos(3:end)  +  "_true_pos_0_" + true_pos(3:end) + ...
            "_C_" + num2str(C(1,2)) + "_samples_" + num2str(N) + "_" + num2str(N_f) + "_Wc_" + num2str(Wc) +  "_l_" + num2str(lt) +"_nel_" + num2str(nel),"SVM") 
    end 

    if count == 60
        break
    end

end