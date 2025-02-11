kor = imread('kor.jpg');
kor_kor = kor;
[row col color] = size(kor_kor);

% image(kor(2000:3648,1:1000,:))
% kor(3000,1,:)

buffer = uint8(0);

for i = 1:row
    for j = 1:col
        
%         only remain pure black pixels with RGB=[0,0,0]
%         if kor_kor(i,j,1) > uint8(20) || kor_kor(i,j,2) > uint8(20) || kor_kor(i,j,3) > uint8(20)
%             kor_kor(i,j,1) = uint8(255);
%             kor_kor(i,j,2) = uint8(255);
%             kor_kor(i,j,3) = uint8(255);

%         change the R-value and B-value of the pixels if R-value less than
%         B-value
        if kor_kor(i,j,1) < kor_kor(i,j,3)
            buffer = kor_kor(i,j,3);
            kor_kor(i,j,1) = kor_kor(i,j,3);
            kor_kor(i,j,3) = buffer;
        end
        
        if kor_kor(i,j,1) > uint8(20) || kor_kor(i,j,2) > uint8(20) || kor_kor(i,j,3) > uint8(20)
            kor_kor(i,j,1) = 1.0 * kor_kor(i,j,1);
            kor_kor(i,j,2) = 1.0 * kor_kor(i,j,2);
            kor_kor(i,j,3) = 0.3 * kor_kor(i,j,3);
        end
            
%         kor_kor(i,j,2) = kor_kor(i,j,2) + 0.5 * (uint8(255) - kor_kor(i,j,1));
%         kor_kor(i,j,1) = kor_kor(i,j,2) + 0.5 * (uint8(255) - kor_kor(i,j,1));

%             if kor_kor(i,j,1) < uint8(128)
%                 kor_kor(i,j,1) = 1.5 * kor_kor(i,j,1);
%             end
            
%         change the R-value and B-value of the pixels if R-value larger than
%         B-value
%         if kor_kor(i,j,1) > kor_kor(i,j,3)
%             buffer = kor_kor(i,j,1);
%             kor_kor(i,j,1) = kor_kor(i,j,3);
%             kor_kor(i,j,3) = buffer;
%         end

 
    end
end

subplot(1,2,1), image(kor)
subplot(1,2,2), image(kor_kor)



imwrite(kor_kor, 'kor_kor.jpg')


