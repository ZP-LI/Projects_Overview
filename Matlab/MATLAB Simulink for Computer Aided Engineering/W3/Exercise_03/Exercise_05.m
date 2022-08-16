clear; close all; clc
%% Subtask 1
K_T = 3.45;
T = 0.001;
i_1 = 1/6.5;
i_2 = 1/5.6;
d = 0.038;
G_wheel = K_T * feedback(tf(1/T, [1 0]), 1) * i_1 * i_2 * d/2;

%% Subtask 2
% Interactive: pidTuner(G_wheel) -> Kp = 5, Ki = 5000

% Programmed:
pidtune(G_wheel, 'pi', 9, pidtuneOptions('PhaseMargin', 90))
% Kp = 5, Ki = 5e+03
