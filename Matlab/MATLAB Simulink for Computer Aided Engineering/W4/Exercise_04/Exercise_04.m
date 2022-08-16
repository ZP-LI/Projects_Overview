clear; close all; clc;
%% Subtask 01
syms x
[solve_x,par,condi] = solve(abs(exp(x)) == 1, 'ReturnConditions', true)
% RESULT: solve_x = y*1i, par = y, condi = in(y, 'real')

%% Subtask 02
assume(x, 'real');
solve(abs(exp(x)) == 1, 'ReturnConditions', true)
% RESULT: 0

%% Subtask 03
syms a b c
S = solve(a^2 * b^2 == 0, a - b/2 == c, [a,b]);
S.a
S.b
% RESULT: S.a = c, 0; S.b = 0, -2*c

%% Subtask 04
assume(c > 1)
assume(b >= 0)
S = solve(a^2*b^2 == 0, a - b/2 == c, [a,b]);
S.a
S.b
% RESULT: S.a = c; S.b = 0