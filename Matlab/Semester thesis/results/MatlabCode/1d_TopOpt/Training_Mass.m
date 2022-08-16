close all
clc
clear all 


%Parameters
nel = 10;
lt = 300; % mm
Wc = 40; % mm


file = load("Figuren_1110\Samples_Mass\samples_8125_Wc_40_l_300_nel_10.mat");


X = file.samples(1:4,:);
y = file.samples(5,:);

N = length(y);

figure(1)
subplot(3,3,1)
scatter(X(1,:),X(2,:),[],y,'filled')
xlabel('k11'); ylabel('k22');
ax = gca; cb = colorbar; colormap(jet(256)); hold on
subplot(3,3,2)
scatter(X(1,:),X(3,:),[],y,'filled')
xlabel('k11'); ylabel('k33');
ax = gca; cb = colorbar; colormap(jet(256)); hold on
subplot(3,3,3)
scatter(X(1,:),X(4,:),[],y,'filled')
xlabel('k11'); ylabel('k66');
ax = gca; cb = colorbar; colormap(jet(256)); hold on
subplot(3,3,4)
scatter(X(2,:),X(3,:),[],y,'filled')
xlabel('k22'); ylabel('k33');
ax = gca; cb = colorbar; colormap(jet(256)); hold on
subplot(3,3,5)
scatter(X(2,:),X(4,:),[],y,'filled')
xlabel('k22'); ylabel('k66');
ax = gca; cb = colorbar; colormap(jet(256)); hold on
subplot(3,3,7)
scatter(X(3,:),X(4,:),[],y,'filled')
xlabel('k33'); ylabel('k66');
ax = gca; cb = colorbar; colormap(jet(256));

%% Mass

if true
%Training

rmax=0;
rmin=1; 
rmean=0; 
NN=8;

for ii=1:NN 
    r = 0.5;
    mserror = 1;
    p=max(y);
    relmaxerr = 1;

    nHl = 1;  % number of hidden layers 1,2,3,4,5
    for i=1:5
        nN = 1;  % number of nodes 1,3,5,7,9,11,13,15,17,19
        for j=1:10
            net = feedforwardnet([nN*ones(1,nHl)]); % hidden layer size
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.1;
            net.divideParam.testRatio = 0.1;
            net.performParam.normalization = 'percent';
            [net, tr] = trainlm(net,X,y);
            [r_new,~,~] = regression(y(tr.testInd), net(X(:,tr.testInd))); % regression value
            p_new = mae(y(tr.testInd)-net(X(:,tr.testInd)));
            mserror_new =  tr.best_tperf; 
            relmaxerr_new = max(abs((y-net(X))./y));
           
            if r < r_new  && mserror > mserror_new
                mserror = mserror_new;
                relmaxerr =relmaxerr_new;
                r = r_new;
                p = p_new;
                result_net = net;
                results_nN = nN;
                results_nHl = nHl;
            end
            nN = nN +2;
        end 
        nHl = nHl + 1;
    end
    
    results_nN
    results_nHl
    net = result_net;
    rvalue = num2str(r);

    save("Figuren_1110/Ratio01_2/Mass_Estimator/ANN_r_" + "0_" + rvalue(3:end) + "_samples_" + num2str(N) + "_Wc_" + num2str(Wc) +  "_l_" + num2str(lt) +"_nel_" + num2str(nel),"net")


end 
    
end 

