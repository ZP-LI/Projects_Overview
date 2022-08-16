clear; close all; clc;
%% ========================================================================
%  SubTask 2a
%  ------------------------------------------------------------------------
global ds

ds = datastore('C:\Users\LENOVO\Desktop\Praktikum MATLAB Simulink for Computer Aided Engineering\W2\Data\Data\RKI_COVID19.csv','TreatAsMissing','NA','MissingValue',0)
ds.Delimiter = ',';

%% ========================================================================
%  SubTask 2b
%  ------------------------------------------------------------------------
ds.VariableNames(3) = {'State'};
ds.SelectedFormats(3) = {'%q'};

ds.VariableNames(5) = {'Age Group'};
ds.SelectedFormats(5) = {'%q'};

ds.VariableNames(6) = {'Sex'};
ds.SelectedFormats(6) = {'%q'};

ds.VariableNames(7) = {'Cases'};
ds.SelectedFormats(7) = {'%f'};

ds.VariableNames(8) = {'Deaths'};
ds.SelectedFormats(8) = {'%f'};

data = preview(ds)
%% ========================================================================
%  SubTask 2c
%  ------------------------------------------------------------------------
t = tall(ds);

%% ========================================================================
%  SubTask 2d
%  ------------------------------------------------------------------------
B = head(t);

%% ========================================================================
%  SubTask 2e
%  ------------------------------------------------------------------------
[Cases, Deaths] = gather(B.Cases, B.Deaths);
sum = sum(Cases) + sum(Deaths);

Group_State = groupsummary(B,"State");

%% ========================================================================
%  SubTask 2f
%  ------------------------------------------------------------------------
State = gather(B.State);
Cases_SH = 0;
Deaths_SH = 0;

for i = 1:8
    if State{i} == 'Schleswig-Holstein'
        Cases_SH = Cases_SH + Cases(i);
        Deaths_SH = Deaths_SH + Deaths(i);
    end
end

figure('Name', 'Pie plot *2')
subplot(1,2,1)
pie(Cases_SH, 'Schleswig-Holstein');
title('Cases in each state in Germany');
subplot(1,2,2)
pie(Deaths_SH, 'Schleswig-Holstein');
title('Deaths in each state in Germany');

%% ========================================================================
%  SubTask 2g_i
%  ------------------------------------------------------------------------
[num, txt, raw] = xlsread('C:\Users\LENOVO\Desktop\Praktikum MATLAB Simulink for Computer Aided Engineering\W2\Data\Data\GPS_Coordinates_StatesGermany.xlsx');

g_State = raw(2:end,1);
g_Lat = cell2mat(raw(2:end, 2));
g_Lon =cell2mat(raw(2:end, 3));

ds.SelectedVariableNames = {'State', 'Cases', 'Deaths'};
ds.ReadSize = 5000;

%% ========================================================================
%  SubTask 2g_ii
%  ------------------------------------------------------------------------
StateNum = length(raw) - 1;
g_sum_cases = zeros(StateNum,1);
g_sum_Deaths = zeros(StateNum,1);

for i = 1:StateNum
    Input_State = g_State{i};
    [g_sum_cases(i), g_sum_Deaths(i)] = datasearch(Input_State);
end

bub_table = table(g_State, g_Lat, g_Lon, g_sum_cases, g_sum_Deaths);

%% ========================================================================
%  SubTask 2g_iii
%  ------------------------------------------------------------------------
figure('Name', 'Bubble figure');
geobubble(g_Lat, g_Lon, g_sum_cases);
