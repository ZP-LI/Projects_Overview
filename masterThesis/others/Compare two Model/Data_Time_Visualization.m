close all
clear
clc

%% Load Data files
data_Alex_067 = readlines("TestAlexModel_067_Li.txt");
data_Alex_080 = readlines("TestAlexModel_080_Li.txt");
data_Alex_100 = readlines("TestAlexModel_100_Li.txt");
data_Alex_125 = readlines("TestAlexModel_125_Li.txt");
data_Yuhong_067 = readlines("TestYuhongModel_067_Li.txt");
data_Yuhong_080 = readlines("TestYuhongModel_080_Li.txt");
data_Yuhong_100 = readlines("TestYuhongModel_100_Li.txt");
data_Yuhong_125 = readlines("TestYuhongModel_125_Li.txt");

%% Data split
delimiter_text = "All kind of Timer:";
delimiter_index = zeros(8, 1);
delimiter_index(1) = find(data_Alex_067 == delimiter_text);
delimiter_index(2) = find(data_Alex_080 == delimiter_text);
delimiter_index(3) = find(data_Alex_100 == delimiter_text);
delimiter_index(4) = find(data_Alex_125 == delimiter_text);
delimiter_index(5) = find(data_Yuhong_067 == delimiter_text);
delimiter_index(6) = find(data_Yuhong_080 == delimiter_text);
delimiter_index(7) = find(data_Yuhong_100 == delimiter_text);
delimiter_index(8) = find(data_Yuhong_125 == delimiter_text);

data_Timer = strings(length(data_Alex_067(delimiter_index(1)+1 : end)), 8);
data_Timer(:, 1) = data_Alex_067(delimiter_index(1)+1 : end);
data_Timer(:, 2) = data_Alex_080(delimiter_index(2)+1 : end);
data_Timer(:, 3) = data_Alex_100(delimiter_index(3)+1 : end);
data_Timer(:, 4) = data_Alex_125(delimiter_index(4)+1 : end);
data_Timer(:, 5) = data_Yuhong_067(delimiter_index(5)+1 : end);
data_Timer(:, 6) = data_Yuhong_080(delimiter_index(6)+1 : end);
data_Timer(:, 7) = data_Yuhong_100(delimiter_index(7)+1 : end);
data_Timer(:, 8) = data_Yuhong_125(delimiter_index(8)+1 : end);

%% Parameter Definition and Assignment
array_Time = strings(size(data_Timer));
label_Time = strings(size(data_Timer, 1), 1);
for i = 1:size(array_Time, 1)
    for j =1:size(array_Time, 2)
        content = split(data_Timer(i, j), ':');
        inter_para = strings(1, 2);
        inter_para(1, :) = content;
        inter_para(1, 1) = strtrim(inter_para(1, 1));
        label_Time(i, 1) = inter_para(1, 1);
        array_Time(i, j) = inter_para(1, 2);
    end
end
array_Time(6, :) =[];

%% Plot Time
array_time = str2double(array_Time(:))';
array_time = reshape(array_time, [5, 8])';
label_time = {label_Time(1, 1), label_Time(2, 1), label_Time(3, 1), label_Time(4, 1), label_Time(5, 1)}';
label_name = {'0.67Hz'; '0.80Hz'; '1.00Hz'; '1.25Hz'};

subplot(2, 1, 1);
h_time1 = bar(array_time(1:4, :));
title('Alex''s Model');
set(h_time1, {'DisplayName'}, label_time);
set(gca, 'xticklabel', label_name);
legend();
for i = 1:size(array_time, 2)
    xtips = h_time1(i).XEndPoints;
    ytips = h_time1(i).YEndPoints;
    labels = string(round(h_time1(i).YData, 2));
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

subplot(2, 1, 2);
h_time2 = bar(array_time(5:8, :));
title('Yuhong''s Model');
set(h_time2, {'DisplayName'}, label_time);
set(gca, 'xticklabel', label_name);
for i = 1:size(array_time, 2)
    xtips = h_time2(i).XEndPoints;
    ytips = h_time2(i).YEndPoints;
    labels = string(round(h_time2(i).YData, 2));
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

sgtitle('Time Graph');

%%