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

%%
frame = frames{1};
[row, col, rgb] = size(frame);

for r = 1:row
    for c = 1:col
        
       if (frame(r, c, 1) < uint8(220)) && (frame(r, c, 2) < uint8(240)) && (frame(r, c, 3) < uint8(220))
           frame(r, c, 1) = uint8(0);
           frame(r, c, 2) = uint8(0);
           frame(r, c, 3) = uint8(0);
        end
    end
end
imshow(frame)
