clear; close all; clc;
%% Subtask 1
H = filt([5 4 3 2 1], 15, 0.02);
%H = tf([5 4 3 2 1], [15 0 0 0 0], 0.02);

%% Subtask 2
H_us = upsample(H, 2);

%% Subtask 3
H_rs = d2d(H, 0.01, 'tustin');

%% Subtask 4
figure(1)
subplot(1,3,1)
step(H); title('H');
subplot(1,3,2)
step(H_us); title('H_us');
subplot(1,3,3)
step(H_rs); title('H_rs');
% System H and System H_us have the same step response.

%% Subtask 5
Y = zpk([ ], [-10, -3, -1, -1], 1);

%% Subtask 6
[Yb,g] = balreal(Y);

%% Subtask 7
Yr = modred(Yb,[3,4]);

%% Subtask 8
figure(2)
step(Y);
hold on
step(Yr);
legend('Y', 'Yr');
hold off
% At first a little different

%% Subtask 9
[Ys,Yf] = freqsep(Y, 2);

%% Subtask 10
figure(3)
step(Ys);
hold on
step(Yf);
step(Ys+Yf);
legend('Ys', 'Yf', 'Ys+Yf');
hold off