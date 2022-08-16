clear; close all; clc;
%% Subtask 1
w = realp('w',3);
z = realp('z',0.9);

%% Subtask 2
Q = ss([-2*w*z, -w^2; 1, 0], [w^2; 0], [0, 1], 0);

%% Subtask 3
Qsample = replaceBlock(Q, 'z', [0.1, 0.7, 1]);

%% Subtask 4
figure(1)
bode(Qsample);

%% Subtask5
figure(2)
pzmap(Qsample(:, :, 1, 3))

%% Subtask 6
damp(Qsample(:, :, 1, 2))
% Damping: 7.00e-01; Frequency: 3.00e+00

%% Subtask 7
20 * log10(getPeakGain(Qsample(:, :, 1, 1), 0.5))
% 14.0197

%% Subtask 8
t = linspace(0, 40, 4000);
input = sin((t/10) .* t);
figure(3)
plot(t, input);

%% Subtask 9
figure(4)
lsim(Qsample(:, :, 1, 1), input, t)
%  Amplitude about 0.1