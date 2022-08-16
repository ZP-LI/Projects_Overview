clear; close all; clc;
%% Subtask 01
syms phi(t) g l m d
dphi = diff(phi,t);
S = dsolve(diff(phi,t,2) == -g*phi/l - d*dphi/(m*l^2), phi(0) == 1, dphi(0) == 0);

%% Subtask 02
S_r = subs(S, [l, m, g, d], [10, 5, 9.81, 50]);

%% Subtask 03
vpa(subs(S_r, t, 5), 40)
% RESULT: 0.1419576473814292547603698105879010822261

%% Subtask 04
fplot3(10*sin(S_r), t, -10*cos(S_r), [0 40])
% PROBLEM: The image is not continuous.