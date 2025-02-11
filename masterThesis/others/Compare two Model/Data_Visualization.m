close all
clear
clc
%% Load Data files
data_all = readlines("第二轮/TestAlexModel_125_TestYuhongTraj_Li.txt"); %%%%%%

%% Data split
delimiter_text = ["Body Position:",
    "Real and Target EndpointPosition of Leg1 (fl):",
    "Real and Target EndpointPosition of Leg2 (fr):",
    "Real and Target EndpointPosition of Leg3 (rl):",
    "Real and Target EndpointPosition of Leg4 (rr):",
    "All kind of Timer:"];
delimiter_index = zeros(length(delimiter_text), 1);
for i = 1:length(delimiter_index)
    delimiter_index(i) = find(data_all == delimiter_text(i));
end

data_BodyPos = data_all(delimiter_index(1)+1 : delimiter_index(1)+10000);
data_Leg1 = data_all(delimiter_index(2)+1 : delimiter_index(2)+10000);
data_Leg2 = data_all(delimiter_index(3)+1 : delimiter_index(3)+10000);
data_Leg3 = data_all(delimiter_index(4)+1 : delimiter_index(4)+10000);
data_Leg4 = data_all(delimiter_index(5)+1 : delimiter_index(5)+10000);
data_Timer = data_all(delimiter_index(6)+1 : end);

%% Parameter Definition
array_BodyPos = zeros(length(data_BodyPos), 3);
array_Leg1 = zeros(length(data_Leg1), 4);
array_Leg2 = zeros(length(data_Leg2), 4);
array_Leg3 = zeros(length(data_Leg3), 4);
array_Leg4 = zeros(length(data_Leg4), 4);
array_Time = strings(length(data_Timer), 2);

%% Parameter Assignment
for i = 1:length(data_BodyPos)
    array_BodyPos(i, :) = split(data_BodyPos(i), ',')';

    endpoint_pos = split(data_Leg1(i), ' ');
    array_Leg1(i, 1:2) = split(endpoint_pos(1), ',')';
    array_Leg1(i, 3:4) = split(endpoint_pos(4), ',')';

    endpoint_pos = split(data_Leg2(i), ' ');
    array_Leg2(i, 1:2) = split(endpoint_pos(1), ',')';
    array_Leg2(i, 3:4) = split(endpoint_pos(4), ',')';

    endpoint_pos = split(data_Leg3(i), ' ');
    array_Leg3(i, 1:2) = split(endpoint_pos(1), ',')';
    array_Leg3(i, 3:4) = split(endpoint_pos(4), ',')';

    endpoint_pos = split(data_Leg4(i), ' ');
    array_Leg4(i, 1:2) = split(endpoint_pos(1), ',')';
    array_Leg4(i, 3:4) = split(endpoint_pos(4), ',')';
end

%% Parameter Assignment 2
for i =1:length(data_Timer)
    content = split(data_Timer(i), ':');
    array_Time(i, :) = content;
    array_Time(i, 1) = strtrim(array_Time(i, 1));
end

%% Plot BodyPosition
t = linspace(0, length(array_BodyPos)*0.002, length(array_BodyPos));

figure('Name', 'BodyPosition');
subplot(3, 1, 1);
plot(t, array_BodyPos(:, 1));
title('x Coordination');
subplot(3, 1, 2);
plot(t, array_BodyPos(:, 2));
title('y Coordination');
subplot(3, 1, 3);
plot(t, array_BodyPos(:, 3));
title('z Coordination');

sgtitle('2nd: BodyPostion of AlexModel with 1.25hz'); %%%%%%

saveas(gcf, 'BodyPosition_Alex125_2nd.png'); %%%%%%
close all

%% Plot LegPosition
f_LP = figure('Name', 'LegPosition');
subplot(2, 2, 1);
plot(array_Leg1(:, 1), array_Leg1(:, 2), 'g', 'LineWidth', 2); hold on
plot(array_Leg1(:, 3), array_Leg1(:, 4), 'r--'); hold off
axis([-0.05 0.05 -0.06 -0.03]);
title('Leg1 (fl)');
subplot(2, 2, 3);
plot(array_Leg3(:, 1), array_Leg3(:, 2), 'g', 'LineWidth', 2); hold on
plot(array_Leg3(:, 3), array_Leg3(:, 4), 'r--'); hold off
axis([-0.05 0.05 -0.06 -0.03]);
title('Leg3 (rl)');
subplot(2, 2, 2);
plot(array_Leg2(:, 1), array_Leg2(:, 2), 'g', 'LineWidth', 2); hold on
plot(array_Leg2(:, 3), array_Leg2(:, 4), 'r--'); hold off
axis([-0.05 0.05 -0.06 -0.03]);
title('Leg2 (fr)');
subplot(2, 2, 4);
plot(array_Leg4(:, 1), array_Leg4(:, 2), 'g', 'LineWidth', 2); hold on
plot(array_Leg4(:, 3), array_Leg4(:, 4), 'r--'); hold off
axis([-0.05 0.05 -0.06 -0.03]);
title('Leg4 (rr)');

l_LP = legend('Target Trajectory', 'Real Trajectory');
l_LP.Position = [0.4 0.45 0.2393 0.0774];

sgtitle('2nd: Leg Position of AlexModel with 1.25hz'); %%%%%%
f_LP.Position = [960 318 2240 840];

saveas(gcf, 'LegPosition_Alex125_2nd.png'); %%%%%%
close all

%% Animation Plot Leg Position
cycle_start_x = find(abs(array_Leg1(:,1)+0.0213) < 0.0007); % AlexModel: 0.035 for <1.00hz; 0.0317 for 1.00hz; 0.0263 for 1.25hz; YuhongModel: always 0.03
cycle_start_y = find(abs(array_Leg1(:,2)+0.045) < 0.0007);
[cycle_start, ~] = intersect(cycle_start_x, cycle_start_y);
i = 1;
while i <= length(cycle_start)-1
    if (cycle_start(i+1)-cycle_start(i)) < 100
        cycle_start(i+1) = [];
    else
        i = i + 1;
    end
end
cycle_start = [1; cycle_start];

h_APL = figure('Name', 'Animation Leg Position');
h_APL.Position = [1840 318 1120 840];

for i = 1:length(array_BodyPos)*0.25
    cycle_previous_idx = find(cycle_start<=i);
    cycle_current_idx = cycle_previous_idx(end);
    cycle_start_timestep = cycle_start(cycle_current_idx);
    
    subplot(2, 2, 1);
    plot(array_Leg1(cycle_start_timestep:i, 1), array_Leg1(cycle_start_timestep:i, 2), 'go'); hold on
    plot(array_Leg1(cycle_start_timestep:i, 3), array_Leg1(cycle_start_timestep:i, 4), 'r*'); hold off
    xlim([-0.05 0.05]);
    ylim([-0.06 -0.030]);
    title('Leg1 (fl)');

    subplot(2, 2, 2);
    plot(array_Leg2(cycle_start_timestep:i, 1), array_Leg2(cycle_start_timestep:i, 2), 'go'); hold on
    plot(array_Leg2(cycle_start_timestep:i, 3), array_Leg2(cycle_start_timestep:i, 4), 'r*'); hold off
    xlim([-0.05 0.05]);
    ylim([-0.06 -0.030]);
    title('Leg2 (fr)');

    subplot(2, 2, 3);
    plot(array_Leg3(cycle_start_timestep:i, 1), array_Leg3(cycle_start_timestep:i, 2), 'go'); hold on
    plot(array_Leg3(cycle_start_timestep:i, 3), array_Leg3(cycle_start_timestep:i, 4), 'r*'); hold off
    xlim([-0.05 0.05]);
    ylim([-0.06 -0.030]);
    title('Leg3 (rl)');

    subplot(2, 2, 4);
    plot(array_Leg4(cycle_start_timestep:i, 1), array_Leg4(cycle_start_timestep:i, 2), 'go'); hold on
    plot(array_Leg4(cycle_start_timestep:i, 3), array_Leg4(cycle_start_timestep:i, 4), 'r*'); hold off
    xlim([-0.05 0.05]);
    ylim([-0.06 -0.030]);
    title('Leg4 (rr)');

    sgtitle('2nd: Animation Leg Position of AlexModel with 1.25hz'); %%%%%%

    a_APL = annotation('textbox', [.48 .32 .1 .2], 'String', num2str(i), 'FitBoxToText', 'on');
    a_APL.FontSize = 20;
    
    drawnow

    frame = getframe(1);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im,256);
    if i == 1
        imwrite(imind, cm, 'Animation Leg Movement_Alex125_2nd.gif', 'gif', 'DelayTime', 0.05, 'Loopcount', inf); %%%%%%
    else
        imwrite(imind, cm, 'Animation Leg Movement_Alex125_2nd.gif', 'gif', 'DelayTime', 0.05, 'WriteMode', 'append'); %%%%%%
    end
    
    delete(a_APL);
end

close all

%%


