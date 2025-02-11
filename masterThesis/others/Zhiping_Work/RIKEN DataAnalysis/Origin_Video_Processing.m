clear
clc
close all

%% Read a video
% file = "Data_20220227\Data_1973DLC\MVI_1973DLC_resnet50_spinalMotionsJan27shuffle1_200000_labeled.mp4"; % Video-File Name
% file = "Data_20220227\Data_1978DLC\MVI_1978DLC_resnet50_spinalMotionsJan27shuffle1_200000_labeled.mp4"; % Video-File Name
file = "Data_20220227\Data_1980DLC\MVI_1980DLC_resnet50_spinalMotionsJan27shuffle1_200000_labeled.mp4"; % Video-File Name
mouse_movie = VideoReader(file);
mouse_movie.CurrentTime = 0;

%% Read every frame
frames = cell(1, mouse_movie.NumFrames);
i = 1;
while hasFrame(mouse_movie)
    
    frames{i} = readFrame(mouse_movie);
    i = i + 1;
    
end

%% Read coordinate information
% Csv_Data = 'Data_20220227\Data_1973DLC\MVI_1973DLC_resnet50_spinalMotionsJan27shuffle1_200000.csv';
% Csv_Data = 'Data_20220227\Data_1978DLC\MVI_1978DLC_resnet50_spinalMotionsJan27shuffle1_200000.csv';
Csv_Data = 'Data_20220227\Data_1980DLC\MVI_1980DLC_resnet50_spinalMotionsJan27shuffle1_200000.csv';
Data = csvread(Csv_Data, 3, 0);

%% Create VideoWriter Object
% v = VideoWriter('mouse_movie_1973DLC.avi');
% v = VideoWriter('mouse_movie_1978DLC.avi');
v = VideoWriter('mouse_movie_1980DLC.avi');

%% Processing every image/frame
N_Len = [801, 1920]; % Required length range, min = 1, max = 1920
col = N_Len(2) - N_Len(1) + 1;
N_Wid = [61, 1080]; % Required width range, min = 1, max = 1080
row = N_Wid(2) - N_Wid(1) + 1;

D_Radius = round(0.5 * min((N_Len(2) - N_Len(1)), (N_Wid(2) - N_Wid(1)))); % Display Radius
D_Center = [row - D_Radius, col - D_Radius]; % Display Center, maybe need to be addied 1

% text_row = 970; % Best entered manually
% text_col = 1070; % Same as above
text_row = 50; % Best entered manually
text_col = 50; % Same as above

open(v);

for i = 1:length(frames)
    
    frame = frames{i};
    frame = frame(N_Wid(1):N_Wid(2), N_Len(1):N_Len(2), :);
    
    for r = 1:row
        for c = 1:col
            if sqrt((r - D_Center(1))^2 + (c - D_Center(2))^2) > D_Radius
                
                frame(r, c, :) = [0, 0, 0];
                
            end
        end
    end
    
    frame = insertText(frame, [text_row, text_col], [':' num2str(i/mouse_movie.FrameRate, '%.0f')], 'TextColor', 'white', 'BoxColor', 'black', 'FontSize', 20);

    for ii = 1:14
    
        idx = 1 + 3 * (ii - 1) + 1;
        p1 = [Data(i, idx) - N_Len(1), Data(i, idx + 1) - N_Wid(1)];
        p2 = [Data(i, idx + 3) - N_Len(1), Data(i, idx + 4) - N_Wid(1)];
               
        if (p1(1) > 0) && (p1(2) > 0) && (p2(1) > 0) && (p2(2) > 0)
            
            if ii == 1
                Distance = sqrt(((p1(1) - p2(1))^2 + (p1(2) - p2(2))^2));
                Tri_1 = [52 / Distance * (p1(1) - p2(1)) + p1(1), 52 / Distance * (p1(2) - p2(2)) + p1(2)];
                Tri_2 = [p1(1) - 30 / Distance * (p1(2) - p2(2)), p1(2) + 30 / Distance * (p1(1) - p2(1))];
                Tri_3 = [p1(1) + 30 / Distance * (p1(2) - p2(2)), p1(2) - 30 / Distance * (p1(1) - p2(1))];
                frame = insertShape(frame, 'Polygon', [Tri_1(1) Tri_1(2) Tri_2(1) Tri_2(2) Tri_3(1) Tri_3(2)], 'LineWidth', 5, 'Color', 'red');
            end

            frame = insertShape(frame, 'Line', [p1(1) p1(2) p2(1) p2(2)], 'LineWidth', 10, 'Color', 'blue');
            frame = insertShape(frame, 'FilledCircle', [p1(1) p1(2) 8; p2(1) p2(2) 8], 'Color', 'red');
            frame = insertShape(frame, 'FilledCircle', [p1(1) p1(2) 3; p2(1) p2(2) 3], 'Color', 'white');

        end
    end
    
    imshow(frame)
    
    writeVideo(v, frame);
    
end

close(v)
% frame = frames{1};
% frame = frame(N_Wid(1):N_Wid(2), N_Len(1):N_Len(2), :);
% 
% for r = 1:row
%     for c = 1:col
%         if sqrt((r - D_Center(1))^2 + (c - D_Center(2))^2) > D_Radius
% 
%             frame(r, c, :) = [0, 0, 0];
% 
%         end
%     end
% end
% 
% frame = insertText(frame, [row - 50, col - 40], [':' num2str(30/mouse_movie.FrameRate, '%.0f')], 'TextColor', 'white', 'BoxColor', 'black', 'FontSize', 20);
% 
% imshow(frame)
% hold on
% for ii = 1:14
%     
%     idx = 1 + 3 * (ii - 1) + 1;
%     p1 = [Data(1, idx + 1) - N_Wid(1), Data(1, idx) - N_Len(1)];
%     p2 = [Data(1, idx + 4) - N_Wid(1), Data(1, idx + 3) - N_Len(1)];
%     display([p1 p2])
%     if (p1(1) > 0) && (p1(2) > 0) && (p2(1) > 0) && (p2(2) > 0)
%         
%         plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','r','LineWidth',2);
%         hold on
%         
%     end    
%     
% end


