function SeamImg=findSeamImg(x)
% FINDSEAMIMG finds the seam map from which the optimal (vertical running) 
% seam can be calculated. Input is gradient image found from findEnergy.m.
%
% The indexing can be interpreted as in this example image:
%   [(i-1,j-1)  (i-1,j)  (i-1,j+1)]
%   [(i,j-1)    (i,j)    (i,j+1)  ]
%   [(i+1,j-1)  (i+1,j)  (i+1,j+1)]
%
% Author: Danny Luong
%         http://danluong.com
%
% Last updated: 12/20/07


[rows cols]=size(x);

SeamImg=zeros(rows,cols);
SeamImg(:,1)=x(:,1);

for i=2:cols
    for j=1:rows
        if j-1<1
            SeamImg(j,i)= x(j,i)+min([SeamImg(j,i-1),SeamImg(j+1,i-1)]);
        elseif j+1>rows
            SeamImg(j,i)= x(j,i)+min([SeamImg(j-1,i-1),SeamImg(j,i-1)]);
        else
            SeamImg(j,i)= x(j,i)+min([SeamImg(j-1,i-1),SeamImg(j,i-1),SeamImg(j+1,i-1)]);
        end
    end
end
