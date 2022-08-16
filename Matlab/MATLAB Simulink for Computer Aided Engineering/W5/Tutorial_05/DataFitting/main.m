clear; close all; clc;
%% Load and visualize the dataset (2 Points)

load('taskDataset.mat')
figure(1)
scatter3(x, y, z)
xlabel('x')
ylabel('y')
zlabel('z')

%% Prepare a model structure (2 Points)

% See modelFunction.m
p_guess = [1 1 5];

%% Define the residual function (2 Points)

residual = @(p) z - modelFunction(x, y, p);

%% Fit the model (2 Points)

options = optimoptions('lsqnonlin','Display','iter');
[p,resnorm,residual,exitflag,output] = lsqnonlin(residual, p_guess, [], [], options);

%% Visualize the fit (2 Points)

[X, Y] = ndgrid(0:0.1:10, 0:0.1:10);
Z = modelFunction(X, Y, p);
figure(2)
mesh(X, Y, Z)
