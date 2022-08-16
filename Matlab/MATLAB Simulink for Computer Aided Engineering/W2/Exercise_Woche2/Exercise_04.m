clear; close all; clc;
%% ========================================================================
%  SubTask 1
%  ------------------------------------------------------------------------
h = figure('Name', 'Subplots');
p = gobjects(1,3);
for piter = 1:numel(p)
    p(piter) = subplot(3,1,piter);
end

%% ========================================================================
%  SubTask 2
%  ------------------------------------------------------------------------
url = 'http://www.mister-foley.com/images/races/melbourne_circuit.jpg';
melbourne = webread(url); % Data structure of "melbourne" is uint8 with size 240*320*3
axes(p(1)); subimage(melbourne); axis off


%% ========================================================================
%  SubTask 3
%  ------------------------------------------------------------------------
[l, m, n] = size(melbourne);
for i = 1:l
    for j = 1:m
        if melbourne(i,j,1) ~= 0 || melbourne(i,j,2) ~= 0 || melbourne(i,j,3) < 250
            melbourne(i,j,1) = 255;
            melbourne(i,j,2) = 255;
            melbourne(i,j,3) = 255;
        end
    end
end
axes(p(2)); subimage(melbourne); axis off

Logical_Array = false(l,m);

for i = 1:l
    for j = 1:m
        if melbourne(i,j,1) == 0
            Logical_Array(i,j) = 1;
        end
    end
end
axes(p(3)); spy(Logical_Array); axis off

%% ========================================================================
%  SubTask 4
%  ------------------------------------------------------------------------
[Y, X] = find(Logical_Array == 1);
xy_Datapair = [X Y];
writematrix(xy_Datapair, 'Points_on_the_track.csv','Delimiter','comma');
