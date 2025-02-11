clear
clc
close all

%%
file_number = 1:8;
% for i = 1:1 % 2:2 3:3 ... 8:8
% for i = 1:length(file_number)
    
%     filename1 = ['Dataset\test_freilegen_Mit_P' num2str(i) '.lvm'];
%     filename2 = ['Dataset\test_freilegen_Ohne_P' num2str(i) '.lvm'];
%     filename3 = ['Dataset\test_gespannt_Mit_P' num2str(i) '.lvm'];
%     filename4 = ['Dataset\test_gespannt_Ohne_P' num2str(i) '.lvm'];
filename_test = ['Dataset\test_x_0001.lvm'];

    fileID1 = fopen(filename_test);
    s1 = strings([14000, 1]);
    line1 = 1;
    while ~feof(fileID1)
        s1(line1, 1) = fgetl(fileID1);
        line1 = line1 + 1;
    end
    fclose(fileID1);

%%
%     fileID2 = fopen(filename2);
%     s2 = strings([14000, 1]);
%     line2 = 1;
%     while ~feof(fileID2)
%         s2(line2, 1) = fgetl(fileID2);
%         line2 = line2 + 1;
%     end
%     fclose(fileID2);
% 
%     fileID3 = fopen(filename3);
%     s3 = strings([14000, 1]);
%     line3 = 1;
%     while ~feof(fileID3)
%         s3(line3, 1) = fgetl(fileID3);
%         line3 = line3 + 1;
%     end
%     fclose(fileID3);
% 
%     fileID4 = fopen(filename4);
%     s4 = strings([14000, 1]);
%     line4 = 1;
%     while ~feof(fileID4)
%         s4(line4, 1) = fgetl(fileID4);
%         line4 = line4 + 1;
%     end
%     fclose(fileID4);

    channels1 = find(s1 == "X_Value	Voltage (Trigger)	Voltage_0 (Trigger)	Comment");
    i = 1;
    while i <= length(channels1)
        if s1(channels1(i)+1) == ""
            channels1(i) = [];
        else
            i = i + 1;
        end
    end

%%
    datas1 = zeros(1100, 2*length(channels1));
    for i = 1:length(channels1)
        data = s1((channels1(i)+1:channels1(i)+1100), 1);
        data = strrep(data, ',', '.');
        data = split(data, "	");
        data = data(:, 2:3);
        data = cellfun(@str2double, data);
        datas1(:, 2*i-1:2*i) = data(:, :);
    end

%     channels2 = find(s2 == "X_Value	Voltage (Trigger)	Voltage_0 (Trigger)	Comment");
%     datas2 = zeros(1100, 2*length(channels2));
%     for i = 1:length(channels2)
%         data = s2((channels2(i)+1:channels2(i)+1100), 1);
%         data = strrep(data, ',', '.');
%         data = split(data, "	");
%         data = data(:, 2:3);
%         data = cellfun(@str2double, data);
%         datas2(:, 2*i-1:2*i) = data(:, :);
%     end
% 
%     channels3 = find(s3 == "X_Value	Voltage (Trigger)	Voltage_0 (Trigger)	Comment");
%     datas3 = zeros(1100, 2*length(channels3));
%     for i = 1:length(channels3)
%         data = s3((channels3(i)+1:channels3(i)+1100), 1);
%         data = strrep(data, ',', '.');
%         data = split(data, "	");
%         data = data(:, 2:3);
%         data = cellfun(@str2double, data);
%         datas3(:, 2*i-1:2*i) = data(:, :);
%     end
% 
%     channels4 = find(s4 == "X_Value	Voltage (Trigger)	Voltage_0 (Trigger)	Comment");
%     datas4 = zeros(1100, 2*length(channels4));
%     for i = 1:length(channels4)
%         data = s4((channels4(i)+1:channels4(i)+1100), 1);
%         data = strrep(data, ',', '.');
%         data = split(data, "	");
%         data = data(:, 2:3);
%         data = cellfun(@str2double, data);
%         datas4(:, 2*i-1:2*i) = data(:, :);
%     end

    %%
    figure
    hold on
    for i = 1:1 %length(channels1) % 1:1, 1:2
        subplot(1, 2, 1);
%         subplot(2, round(length(channels1)/2), i);
        plot(1:1100, datas1(:, 1*i)); hold on
        subplot(1, 2, 2);
        plot(1:1100, datas1(:, 2*i)); hold on
%         plot(1:1100, datas2(:, 2*i)); hold on
%         plot(1:1100, datas3(:, 2*i)); hold on
%         plot(1:1100, datas4(:, 2*i)); hold on
%         titlename = ['Channel ' num2str(i)];
%         subtitle(titlename);
    end
%     legend('freilegen Mit', 'freilegen Ohne', 'gespannt Mit', 'gespannt Ohne');
    hold off

% end
