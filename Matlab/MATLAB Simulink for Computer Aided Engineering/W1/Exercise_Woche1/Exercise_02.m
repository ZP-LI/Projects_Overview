clear; close all; clc;
%% ========================================================================
%  SubTask 1
%  ------------------------------------------------------------------------
CellData = {'Markus', 'Mueller', 53, true, {'Marta Mueller'; 'Michael Mueller'; 'Martina Mueller'};
                   'Peter', 'Schmidt', 58, true, {'Ursula Schmidt'};
                   'Beate', 'Maier', 46, false, {'Gustav Maier'};
                   'Ursula', 'Leitner', 36, true, {}};

%% ========================================================================
%  SubTask 2
%  ------------------------------------------------------------------------
save('CellData.mat', 'CellData');

%% ========================================================================
%  SubTask 3
%  ------------------------------------------------------------------------
StructData.FirstName = CellData(:,1);
StructData.Surname = CellData(:,2);
StructData.Age = CellData(:,3);
StructData.Attendance = CellData(:,4);
StructData.Company = CellData(:,5);

%% ========================================================================
%  SubTask 4
%  ------------------------------------------------------------------------
Mueller.FirstName = CellData(1,1);
Mueller.Surname = CellData(1,2);
Mueller.Age = CellData(1,3);
Mueller.Attendance = CellData(1,4);
Mueller.Company = CellData(1,5);
whos

%% ========================================================================
%  SubTask 5
%  ------------------------------------------------------------------------
TableData = struct2table(StructData);
%% ========================================================================
%  SubTask 6
%  ------------------------------------------------------------------------
ArrayAttendance = cell2mat(StructData.Attendance);
ArrayAge = cell2mat(StructData.Age);
meanAge = mean(ArrayAge(ArrayAttendance));
