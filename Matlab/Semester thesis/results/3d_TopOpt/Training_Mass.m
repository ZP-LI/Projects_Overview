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
lz = 25;
nelx = 30; %60
nely = 10; %10 
nelz = 4;
ndofs = 3;
a = 0.49*lx/nelx; %x
b = 0.49*ly/nely; %y
c = 0.49*lz/nelz; %z


file = load("Versuch_7500\Samples_Mass\samples_8371_lx_300_ly_75_lz_25_nelx_30_nely_10_nelz_4_border.mat");


X = file.samples(1:4,:);
y = file.samples(5,:);
N = length(y);


figure(1)
subplot(3,3,1)
scatter(X(1,:),X(2,:),[],y,'filled'); hold on;
xlabel('k11'); ylabel('k22');hold on;
subplot(3,3,2)
scatter(X(1,:),X(3,:),[],y,'filled');hold on;
xlabel('k11'); ylabel('k33');
subplot(3,3,3)
scatter(X(1,:),X(4,:),[],y,'filled');hold on;
xlabel('k11'); ylabel('k66');
subplot(3,3,4)
scatter(X(2,:),X(3,:),[],y,'filled');hold on;
xlabel('k22'); ylabel('k33');
subplot(3,3,5)
scatter(X(2,:),X(4,:),[],y,'filled');hold on;
xlabel('k22'); ylabel('k66');
subplot(3,3,7)
scatter(X(3,:),X(4,:),[],y,'filled');hold on;
xlabel('k33'); ylabel('k66');



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
    
    mserror_ = fix(mserror);
    mserror = mserror - mserror_;
    mserror = num2str(mserror); 
    mserror_ = num2str(mserror_); 

    save("Versuch_7500/Mass_Estimator/ANN_r2_" + "0_" + rvalue(3:end) + "_mse_" + mserror_ + "_" + mserror(3:end) + ...
         "_samples_" + num2str(N) + "_lx_" + num2str(lx) + "_ly_" + num2str(ly) + "_lz_" + num2str(lz) + ...
          "_nelx_" + num2str(nelx) + "_nely_" + num2str(nely) + "_nelz_" + num2str(nelz),"net")


end 
    
end 

