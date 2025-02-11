clear all
clc
close all

%% Load Data
% Csv_Data = '1973DLC/MVI_1973DLC_resnet50_spinalMotionsJan27shuffle1_200000.csv';
% Csv_Data = '1978DLC/MVI_1978DLC_resnet50_spinalMotionsJan27shuffle1_200000.csv';
Csv_Data = '1980DLC/MVI_1980DLC_resnet50_spinalMotionsJan27shuffle1_200000.csv';
Data = csvread(Csv_Data, 3, 0);

%% Plot X-Y Postion
figure(1)
for i = 1:15 
    idx = 1 + 3 * (i - 1) + 1;
    color = i / 15 * 10;
    c = ones(length(Data(:, idx)), 1) * color;
    scatter(Data(:, idx), Data(:, idx + 1), [], c, 'filled')
    hold on
end
hold off
colormap(jet)
colorbar('Ticks', linspace(1, 10, i),...
              'TickLabels', {'head', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6', 'back7', 'back8', 'back9', 'back10', 'back11', 'back12', 'back13', 'back14'});
% title('Trajectory plot 1973DLC')
% title('Trajectory plot 1978DLC')
title('Trajectory plot 1980DLC')
xlabel('X position in pixels')
ylabel('Y position in pixels')

%% Plot Frame Index-X Postion
figure(2)
for i = 1:15
    idx = 1 + 3 * (i - 1) + 1;
    plot(Data(:, 1), Data(:, idx))
    hold on
end
hold off
colormap(jet)
legend({'head', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6', 'back7', 'back8', 'back9', 'back10', 'back11', 'back12', 'back13', 'back14'})
% colorbar('Ticks', linspace(1, 10, i),...
%               'TickLabels', {'head', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6', 'back7', 'back8', 'back9', 'back10', 'back11', 'back12', 'back13', 'back14'});
% title('Trajectory plot 1973DLC')
% title('Trajectory plot 1978DLC')
title('Trajectory plot 1980DLC')
xlabel('Frame Index')
ylabel('X-(dashed) position in pixels')

%% Plot Frame Index-Y Postion
figure(3)
for i = 1:15
    idx = 1 + 3 * (i - 1) + 1;
    plot(Data(:, 1), Data(:, idx + 1))
    hold on
end
hold off
colormap(jet)
legend({'head', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6', 'back7', 'back8', 'back9', 'back10', 'back11', 'back12', 'back13', 'back14'})
% colorbar('Ticks', linspace(1, 10, i),...
%               'TickLabels', {'head', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6', 'back7', 'back8', 'back9', 'back10', 'back11', 'back12', 'back13', 'back14'});
% title('Trajectory plot 1973DLC')
% title('Trajectory plot 1978DLC')
title('Trajectory plot 1980DLC')
xlabel('Frame Index')
ylabel('Y-(solid) position in pixels')

%% Plot Likelihood
figure(4)
for i = 1:15
    idx = 1 + 3 * (i - 1) + 1;
    plot(Data(:, 1), Data(:, idx + 2))
    hold on
end
hold off
colormap(jet)
legend({'head', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6', 'back7', 'back8', 'back9', 'back10', 'back11', 'back12', 'back13', 'back14'})
% colorbar('Ticks', linspace(1, 10, i),...
%               'TickLabels', {'head', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6', 'back7', 'back8', 'back9', 'back10', 'back11', 'back12', 'back13', 'back14'});
% title('Likelihood plot 1973DLC')
% title('Likelihood plot 1978DLC')
title('Likelihood plot 1980DLC')
xlabel('Frame Index')
ylabel('Likelihood (use to set pcutoff)')

%% Animation X-Y Position
figure(5)
% filename = 'Animation_XY_Position_1973DLC.gif';
% filename = 'Animation_XY_Position_1978DLC.gif';
filename = 'Animation_XY_Position_1980DLC.gif';
h = plot(NaN, NaN);
hold on
color = linspace(1, 10, 15);
s = scatter(NaN, NaN, 'filled');
hold off
axis([200 1800 300 1100]);
title('Trajectory plot 1973DLC')
xlabel('X position in pixels')
ylabel('Y position in pixels')
colormap(jet)
colorbar('Ticks', linspace(1, 10, 15),...
                  'TickLabels', {'head', 'back1', 'back2', 'back3', 'back4', 'back5', 'back6', 'back7', 'back8', 'back9', 'back10', 'back11', 'back12', 'back13', 'back14'});
  
for j = 1:length(Data)
    X = ones(15, 1);
    Y = ones(15, 1);
    for i = 1:15
        idx = 1 + 3 * (i - 1) + 1;
        X(i) = Data(j, idx);
        Y(i) = Data(j, idx + 1);
    end
    pause(0.01)
    set(h, 'XData', X, 'YData', Y);
    hold on
    set(s, 'XData', X, 'YData', Y, 'CData', color, 'SizeData', 80);
    hold off
    
    drawnow
    
    frame = getframe(5);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im,256);
    if j == 1
        imwrite(imind, cm, filename, 'gif', 'DelayTime', 0.05, 'Loopcount', inf);
    else
        imwrite(imind, cm, filename, 'gif', 'DelayTime', 0.05, 'WriteMode', 'append');
    end

end















