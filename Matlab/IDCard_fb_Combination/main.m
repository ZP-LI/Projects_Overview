clear
clc
close all

%% Load image
filename1 = "身份证_页1.jpg";
filename2 = "身份证_页2.jpg";

im1 = imread(filename1);
im2 = imread(filename2);

%% Image Processing 1
imshow(im1)
roi_1 = drawpolygon();
x = roi_1.Position(:, 1);
y = roi_1.Position(:, 2);
bw_1 = poly2mask(x, y, size(im1, 1), size(im1, 2));

%% Image Processing 2
imshow(im2)
roi_2 = drawpolygon();
x = roi_2.Position(:, 1);
y = roi_2.Position(:, 2);
bw_2 = poly2mask(x, y, size(im2, 1), size(im2, 2));

%% Combine Image
im_new = uint8(ones(size(im1, 1), size(im1, 2), size(im1, 3))) * 254;
for i = 1:size(im_new, 1)
    for j = 1:size(im_new, 2)

        if bw_1(i, j) == 1
            im_new(i, j, :) = im1(i, j, :);
        end

        if bw_2(i, j) == 1
            im_new(i, j, :) = im2(i, j, :);
        end

    end
end

imshow(im_new)

%% Save Image
imwrite(im_new, '身份证_合成.jpg')

