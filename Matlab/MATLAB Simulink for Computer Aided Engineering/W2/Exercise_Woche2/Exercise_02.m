clear; close all; clc;
%% ========================================================================
%  SubTask 1
%  ------------------------------------------------------------------------
clear; close all; clc;

local_var_count = 0;
global_var_count = 0;

global global_var_count
count = 0;

for count = 1:10
    local_var_count = local_var(local_var_count);
    persistent_var_count = persistent_var;
    global_var;
end

%% ========================================================================
%  SubTask 2
%  ------------------------------------------------------------------------
tic;
x = 0;
for i = 1:1000
    x(i) = i^2;
end
toc

tic;
y = 0;
y = zeros(1,1000);
for i = 1:1000
    y(i) = i^2;
end
toc

%% ========================================================================
%  SubTask 3
%  ------------------------------------------------------------------------
A = magic(20000); % increase of used memory

B = A; % The used memory remains unchanged.

B(1) = 0; % increase of used memory
