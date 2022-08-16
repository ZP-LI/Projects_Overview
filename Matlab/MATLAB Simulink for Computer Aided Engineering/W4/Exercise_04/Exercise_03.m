clear; close all; clc;
%% Subtask 01
syms x
F01(x) = 27 * x * log(x+3) + cos(2^x);
double(int(F01, x, -1, 1))
% RESULT: 7.0068

%% Subtask 02
syms y z
F02(x) = z * tan(y*z) / y + log(cos(y*z)) / (y^2);
simplify(diff(F02, z))
% RESULT: -z/(sin(y*z)^2 - 1)
% QUSETION: How to simplify further?

%% Subtask 03
syms alpha
curl([alpha * z / x; z / y; log(x*y)])
% RESULT: [0; alpha/x - 1/x; 0]

%% Subtask 04
syms a s t
ilaplace(1/sqrt(s+a), s, t)
% RESULT: exp(-a*t)/(t^(1/2)*pi^(1/2))

%% Subtask 05
taylortool(exp(x))
% RESULT: exp(1) + exp(1)*(x - 1) + (exp(1)*(x - 1)^2)/2
taylor(exp(x),x,1,'Order',3)