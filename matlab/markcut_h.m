function x=markcut_h(x,SeamVector,markflag)
% SEAMCUT takes as input a RGB or grayscale image and SeamVector array to
% find the pixels contained in the seam, and to remove them from the image.
% Each col of SeamVector must be a single seam.
%
% Author: Danny Luong
%         http://danluong.com
%
% Last updated: 12/20/07

[rows cols dim]=size(x);
[SVrows SVcols SVdim]=size(SeamVector);

if cols~=SVrows
    error('SeamVector and image dimension mismatch');
end

for k=1:SVcols              %goes through set of seams
        for j=1:cols        %goes through each col in image
            if SeamVector(j,k)==1
                %CutImg(:,j,i)=[x(2:rows,j,i)];
                for ii= 2:rows
                    CutImg(ii-1,j)= x(ii,j);
                end
                if markflag ==1
                    CutImg(1,j)= 1;
                else
                    if CutImg(1,j)==0;
                        CutImg(1,j)= 8;
                    end
                end
            elseif SeamVector(j,k)==rows
                %CutImg(:,j,i)=[x(1:rows-1,j,i)];
                for ii= 1:rows-1
                    CutImg(ii,j)= x(ii,j);
                end
                
                if markflag ==1
                    CutImg(rows-1,j)= 1;
                else
                    if CutImg(rows-1,j)==0;
                        CutImg(rows-1,j)= 2;
                    end
                end
            else
                %CutImg(:,j,i)=[x(1:SeamVector(j,k)-1,j,i) x(SeamVector(j,k)+1:rows,j,i)];
                for ii=1:SeamVector(j,k)-1
                    CutImg(ii,j)= x(ii,j);
                end
                for ij=SeamVector(j,k)+1:rows
                   CutImg(ij-1,j)= x(ij,j);
                end
                if markflag ==1
                    CutImg(SeamVector(j,k)-1,j)= 1;
                    CutImg(SeamVector(j,k),j)= 1;
                else
                    if CutImg(1,j)==0;
                        CutImg(SeamVector(j,k)-1,j)= 2;
                    end
                    if CutImg(SeamVector(j,k),j)==0;
                        CutImg(SeamVector(j,k),j)= 2;
                    end
                end
            end
        end
    x=CutImg;
    clear CutImg
    [rows cols dim]=size(x);
end