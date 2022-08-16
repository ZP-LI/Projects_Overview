clear; close all; clc;
%% Subtask 01
digits_old = digits;

%% Subtask 02
digits(20)

%% Subtask 03
e_exact = sym(str2sym('exp(1)'));

%% Subtask 04
e_approx = sym(exp(1),'d');

%% Subtask 05
double(e_approx - e_exact)
% RESULT: 2.9954e-16

%% Subtask 06
digits(digits_old)