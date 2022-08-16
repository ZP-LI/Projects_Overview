clear; close all; clc;
%% Subtask 1
G = tf(1, [0.25, 1, 1]);

%% Subtask 2
[cont,inf] = pidtune(G, 'p', 2.5)
% phase margin is 77.3196

%% Subtask 3
pidTuner(G)
% Kp = 39, Ki = 78, Kd = 4.9

%% Subtask 4
controlSystemDesigner(G)

%% Subtask 7
% C = 3.3258*(1+s)/(1+s/6)

%% Subtask 9
% Peak: 0.995 at 0.36s
