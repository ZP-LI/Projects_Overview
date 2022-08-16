clear; close all; clc;
%% Subtask 01
syms p q r

%% Subtask 02
K = log(p^q);

%% Subtask 03
subs(K,q,0)
% RESULT: 0

%% Subtask 04
L(r,p,q) = sin(r)^2 + cos(p+q)^2;

%% Subtask 05
L(r,r-q,q)
% RESULT: cos(r)^2 + sin(r)^2

%% Subtask 06
simplify(L(r,r-q,q))
% RESULT: 1