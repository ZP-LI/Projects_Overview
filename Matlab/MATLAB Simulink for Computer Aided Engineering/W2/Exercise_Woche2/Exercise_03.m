clear; close all; clc;
%% ========================================================================
%  SubTask 1
%  ------------------------------------------------------------------------
load('GermanPopulation');
GermanPopulation_Array = table2array(PopulationData);

%% ========================================================================
%  SubTask 2
%  ------------------------------------------------------------------------
figure('Name','SubTask 2');
plot(GermanPopulation_Array(:,1), GermanPopulation_Array(:,9))

%% ========================================================================
%  SubTask 3
%  ------------------------------------------------------------------------
% title('German Population');
% xlabel('Year');
% ylabel('Population');
% grid on

%% ========================================================================
%  SubTask 4
%  ------------------------------------------------------------------------
legend('Total Population');
gca

%% ========================================================================
%  SubTask 5
%  ------------------------------------------------------------------------
title('German Population');
xlabel('Year');
ylabel('Population');
grid on
hold on
plot(GermanPopulation_Array(:,1), GermanPopulation_Array(:,11));
plot(GermanPopulation_Array(:,1), GermanPopulation_Array(:,12));
hold off
legend('Total Population', 'Female Population', 'Male Population');

%% ========================================================================
%  SubTask 6
%  ------------------------------------------------------------------------
figure('Name', 'SubTask 6_Area');
area_x = GermanPopulation_Array(:,1);
area_y = [GermanPopulation_Array(:,11) GermanPopulation_Array(:,12)];
h = area(area_x, area_y);
set(h(1), 'FaceColor', [1 1 0])
set(h(2), 'FaceColor', [0 0 1])
legend('Female', 'Male');

figure('Name', 'SubTask 6_Pie');
pie_Female = sum(GermanPopulation_Array(:,11));
pie_Male = sum(GermanPopulation_Array(:,12));
pie([pie_Female pie_Male], {'Female', 'Male'});
legend('Female', 'Male');

%% ========================================================================
%  SubTask 7
%  ------------------------------------------------------------------------
L = 160*membrane(1,100);
figure('Name', 'SubTask 7');
subplot(2,2,1);
s1 = surface(L);
s1.EdgeColor = 'none';
view(90,0);

subplot(2,2,2);
s2 = surface(L);
s2.EdgeColor = 'none';
view(0,90);

subplot(2,2,3);
s3 = surface(L);
s3.EdgeColor = 'none';
view(0,0);

subplot(2,2,4);
s4 = surface(L);
s4.EdgeColor = 'none';
view(90,90);