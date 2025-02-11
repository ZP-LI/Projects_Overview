clear all
clc
close all

%%
filename_num = {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61'};
filename = strcat('images/yueyue_sum_页面_', filename_num{1}, '.jpg');
image_yue = imread(filename);

image_yue_kor = image_yue;

image_size = size(image_yue_kor);
row = image_size(1);
col = image_size(2);

for r = 1:row
    for c = 1:col
        if image_yue_kor(r, c, 1) > 150 || image_yue_kor(r, c, 2) > 150 || image_yue_kor(r, c, 3) > 150
            image_yue_kor(r, c, :) = [255, 255, 255];
        else
            image_yue_kor(r, c, :) = [0, 0, 0];
        end
    end
end
% subplot(1,2,1)
% imshow(image_yue_kor)
% hold on

%%
% count = [];
% r_center = 1;
% while r_center < row
%     if image_yue(r_center, col/2, 1) == 0
%         count = [count r_center];
%         r_center = r_center + 5;
%     else
%         r_center = r_center + 1;
%         continue
%     end
% end

%%
line = zeros(1, col);
line_num = 1;

half_col = round(col/2);
r_center = 1;
while r_center < row
    if image_yue_kor(r_center, half_col, 1) == 0
        line(line_num, half_col) = r_center;
        r_center = r_center + 5;
    else
        r_center = r_center + 1;
        continue
    end
        
    for c_num = half_col + 1: col
        r_neibor_max = min(line(line_num, c_num - 1) + 3, row);
        r_neibor_min = max(1, line(line_num, c_num - 1) - 3);
        line_buffer = [];
        for r_num = r_neibor_min:r_neibor_max
            if image_yue_kor(r_num, c_num, 1) == 0
                line_buffer = [line_buffer r_num];
            end
        end
        if ~isempty(line_buffer)
            line(line_num, c_num) = line_buffer(round(length(line_buffer) / 2));
        end
    end

    for c_num = half_col - 1:-1: 1
        r_neibor_max = min(line(line_num, c_num + 1) + 3, row);
        r_neibor_min = max(1, line(line_num, c_num + 1) - 3);
        line_buffer = [];
        for r_num = r_neibor_min:r_neibor_max
            if image_yue_kor(r_num, c_num, 1) == 0
                line_buffer = [line_buffer r_num];
            end
        end
        if ~isempty(line_buffer)
            line(line_num, c_num) = line_buffer(round(length(line_buffer) / 2));
        end
    end

    if line(line_num, half_col + 150) > 0
        if line(line_num, half_col - 150) > 0
            if line(line_num, half_col + 300) > 0
                if line(line_num, half_col - 300) > 0
                    line_num = line_num + 1;
                    line(line_num, :) = zeros(1, col);
                end
            end
        end
    end
end
line(end, :) = [];

%%
% [row_line, col_line] = size(line);
% image_grade = uint8(ones(row_line, col_line, 3)) * 255;
% for r=1:row_line
%     for c=1:col_line
%         if line(r,c) > 1
%             image_grade(line(r,c), c, :) =[0, 0, 0];
% %             image_grade(line(r,c) + 2, c, :) =[0, 0, 0];
%             image_grade(line(r,c) + 1, c, :) =[0, 0, 0];
%             image_grade(line(r,c) - 1, c, :) =[0, 0, 0];
% %             image_grade(line(r,c) - 2, c, :) =[0, 0, 0];
%         end
%     end
% end
% subplot(1,2,2)
% imshow(image_grade)

%%
line2 = zeros(row, 1);
line2_num = 1;

half_row = round(row/2);
c_center = 1;
while c_center < col
    if image_yue_kor(half_row, c_center, 1) == 0
        line2(half_row, line2_num) = c_center;
        c_center = c_center + 5;
    else
        c_center = c_center + 1;
        continue
    end
        
    for r2_num = half_row + 1: row
        c_neibor_max = min(line2(r2_num - 1, line2_num) + 3, col);
        c_neibor_min = max(1, line2(r2_num - 1, line2_num) - 3);
        line2_buffer = [];
        for c2_num = c_neibor_min:c_neibor_max
            if image_yue_kor(r2_num, c2_num, 1) == 0
                line2_buffer = [line2_buffer c2_num];
            end
        end
        if ~isempty(line2_buffer)
            line2(r2_num, line2_num) = line2_buffer(round(length(line2_buffer) / 2));
        end
    end

    for r2_num = half_row - 1:-1: 1
        c_neibor_max = min(line2(r2_num + 1, line2_num) + 3, col);
        c_neibor_min = max(1, line2(r2_num + 1, line2_num) - 3);
        line2_buffer = [];
        for c2_num = c_neibor_min:c_neibor_max
            if image_yue_kor(r2_num, c2_num, 1) == 0
                line2_buffer = [line2_buffer c2_num];
            end
        end
        if ~isempty(line2_buffer)
            line2(r2_num, line2_num) = line2_buffer(round(length(line2_buffer) / 2));
        end
    end

    if line2(half_row + 150, line2_num) > 10
        if line2(half_row - 150, line2_num) > 10
            if line2(half_row + 300, line2_num) > 10
                if line2(half_row - 300, line2_num) > 10
                    line2_num = line2_num + 1;
                    line2(:, line2_num) = zeros(row, 1);
                end
            end
        end
    end
end
line2(:, end) = [];

%%
% for r=1:2000
%     for c=1:9
%         if line2(r,c) > 1
%             image_grade(r, line2(r,c), :) =[0, 0, 0];
% %             image_grade(line(r,c) + 2, c, :) =[0, 0, 0];
%             image_grade(r, line2(r,c) + 1, :) =[0, 0, 0];
%             image_grade(r, line2(r,c) - 1, :) =[0, 0, 0];
% %             image_grade(line(r,c) - 2, c, :) =[0, 0, 0];
%         end
%     end
% end
% subplot(1,2,2)
% imshow(image_grade)

%%
size_line = size(line);
row_gridnum = size_line(1);
size_line2 = size(line2);
col_gridnum = size_line2(2);
image_cell = cell(row_gridnum + 1, col_gridnum + 1);
image_cell(:) = {uint8(ones(300, 400, 3)) * 255};
c_cell_internindex_arr = zeros(row, col);
r_cell_internindex_arr = zeros(row, col);

for r=1:row
    for c=1:col
        r_cell_index = 1;
        c_cell_index = 1;
        r_cell_internindex = r;
        c_cell_internindex = c;
        for cc = 1:col_gridnum
            if c >= line2(r, cc)
                c_cell_index = c_cell_index + 1;
                c_cell_internindex = c - line2(r, cc) + 100;
                c_cell_internindex_arr(r, c) = c_cell_internindex;
                if c <= line2(r, cc) + 5 && c >= line2(r, cc) - 5
                    c_cell_internindex = 500;
                end
            else
                break
            end
        end
        for rr = 1:row_gridnum
            if r >= line(rr, c)
                r_cell_index = r_cell_index + 1;
                r_cell_internindex = r - line(rr, c) + 150;
                r_cell_internindex_arr(r, c) = r_cell_internindex;
                if r <= line(rr, c) + 5 && r >= line(rr, c) - 5
                    r_cell_internindex = 500;
                end
            else
                break
            end
        end
        if r_cell_internindex <= 300 && c_cell_internindex <= 400
            image_cell{r_cell_index, c_cell_index}(r_cell_internindex, c_cell_internindex, :) = image_yue_kor(r, c, :);
        end
    end
end
% imshow(image_cell{1,1})

%%
out = cell(row_gridnum + 1, col_gridnum + 1);
ocrResult = '';
for r = 1:row_gridnum + 1
    for c = 1:col_gridnum + 1
        ocrResult = ocr(image_cell{r, c}).Text;
        if ~isempty(ocrResult)
            out{r, c} = ocrResult;
        end
    end
end

%%
% for r = 1:row_gridnum + 1
%     for c = 1:col_gridnum + 1
%         if any(isstrprop(out{r, c}, 'digit'))
%             out{r, c} = str2double(out{r, c});
%         end
%     end
% end
%%
T = cell2table(out);
writetable(T, 'test.xlsx', 'Sheet', str2int(filename_num{1}))

