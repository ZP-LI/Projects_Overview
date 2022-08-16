clear; close all; clc;
%% ========================================================================
%  SubTask 1
%  ------------------------------------------------------------------------
PopulationData = readtable('PopulationData.dat'); % Programmatically: Easy for subsequent operations.

% Interactively: Import Data -> Column delimiters: Comma -> Range A2:J56

%% ========================================================================
%  SubTask 2
%  ------------------------------------------------------------------------
summary(PopulationData)

%% ========================================================================
%  SubTask 3
%  ------------------------------------------------------------------------
PopulationData.Population_Female = PopulationData.Population_Total .* PopulationData.Population_Female__OfTotal_ * 0.01;
PopulationData.Population_Male = PopulationData.Population_Total .* (1 - PopulationData.Population_Female__OfTotal_ * 0.01);

%% ========================================================================
%  SubTask 4
%  ------------------------------------------------------------------------
PopulationData_Array = table2array(PopulationData); % Alternative: PopulationData_Array = PopulationData{:,:};
csvwrite('GermanPopulation.csv', PopulationData_Array); % alternativ command: writematrix

%% ========================================================================
%  SubTask 5
%  ------------------------------------------------------------------------
save('GermanPopulation.mat', 'PopulationData');

%% ========================================================================
%  SubTask 6
%  ------------------------------------------------------------------------
PopulationData_LowLv = fopen('PopulationData.dat');
[row, column] = size(PopulationData); % In Subtask 3 there are two added new columns
Header = textscan(PopulationData_LowLv, '%s', column - 2, 'Delimiter', ';');
Numeric_Array = [ ];
for i = 1:row
    Numeric_Array_Row = fscanf(PopulationData_LowLv, ['%f'  ';']);
    Numeric_Array = [Numeric_Array Numeric_Array_Row];
end
Numeric_Array = Numeric_Array';
fclose(PopulationData_LowLv);

%% ========================================================================
%  SubTask 7
%  ------------------------------------------------------------------------
fprintf('%d %.3f %.3f %.3f %.3f %.3f %.3f %.3f %d %.3f \n', Numeric_Array')
