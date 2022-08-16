clear; close all; clc;
%% Subtask 01
syms w_r(t) w_l(t) d l_w
assume(t, 'positive')
v_r(t) = w_r(t) * d/2;
v_l(t) = w_l(t) * d/2;
v_for(t) = (v_r(t) + v_l(t)) / 2;
w_for(t) = (v_r(t) - v_l(t)) / l_w;

%% Subtask 02
v_star = subs(v_for(t), [w_l(t), w_r(t)], [8*sin(t)^2, 8*cos(t)^2])
w_star = subs(w_for(t), [w_l(t), w_r(t)], [8*sin(t)^2, 8*cos(t)^2])

v_star = simplify(v_star)
w_star = simplify(w_star)
% RESULT: v_star = 2*d*sin(t)^2 + 2*d*cos(t)^2; 
%               w_star = -(4*d*sin(t)^2 - 4*d*cos(t)^2)/l_w
% Simplify: v_star = 2*d; w_star = (4*d*cos(2*t))/l_w;

%% Subtask 03
fplot(subs(w_star, [l_w, d], [0.1, 0.03]))
vpasolve(subs(w_star, [l_w, d], [0.1, 0.03]), [0,1])
% RESULT: c.a. 0.785

%% Subtask 04
psi = int(subs(w_star, [l_w, d], [0.1, 0.03]), t);
x = double(int(subs(v_star, d, 0.03) * cos(psi), t, 0, 50))
y = double(int(subs(v_star, d, 0.03) * sin(psi), t, 0, 50))
% RESULT: x = 2.7349, y = 0.0025