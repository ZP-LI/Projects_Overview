clear; close all; clc;
%% Subtask 1
F = zpk(18, [-3-2i, -3+2i], 4/2225);

%% Subtask 2
G = -4/2225 * tf ([1 -23], [1 1 4.25], 'InputDelay', 0.05);

%% Subtask 3
G_zpk = zpk(G);

% Numerator of G_zpk: -0.0017978(s-23)

%% Subtask 4 
E = [F; parallel(F,G)];

%% Subtask 5
C = pid(3.16, 15.9, 0.156);

%% Subtask 6
CE = series(C, E);
closed_loop = feedback(CE, 1, 1, 2);

%% Subtask 7
order(closed_loop)

%% Subtask 8
isproper(closed_loop(2,1))