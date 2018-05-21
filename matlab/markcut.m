function x=markcut(x,SeamVector,markflag)
% SEAMCUT takes as input a RGB or grayscale image and SeamVector array to
% find the pixels contained in the seam, and to remove them from the image.
% Each col of SeamVector must be a single seam.
%
% Author: Danny Luong
%         http://danluong.com
%
% Last updated: 12/20/07


[rows cols]=size(x);
[SVrows SVcols SVdim]=size(SeamVector);

if rows~=SVrows
    error('SeamVector and image dimension mismatch');
end

for k=1:SVcols              %goes through set of seams
        for j=1:rows        %goes through each row in image
            if SeamVector(j,k)==1
                CutImg(j,:)=[x(j,2:cols)];
                if markflag ==1
                    CutImg(j,1)= 1;
                else
                    if CutImg(j,1)==0;
                        CutImg(j,1)= 2;
                    end
                end
            elseif SeamVector(j,k)==cols
                CutImg(j,:)=[x(j,1:cols-1)];
                if markflag ==1
                    CutImg(j,cols-1)= 1;
                else
                    if CutImg(j,cols-1)==0;
                        CutImg(j,cols-1)= 2;
                    end
                end
            else
                CutImg(j,:)=[x(j,1:SeamVector(j,k)-1) x(j,SeamVector(j,k)+1:cols)];
                if markflag ==1
                    CutImg(j,SeamVector(j,k)-1)=1;
                    CutImg(j,SeamVector(j,k))=1;
                else
                    if CutImg(j,SeamVector(j,k)-1)==0;
                        CutImg(j,SeamVector(j,k)-1)= 2;
                    end
                    if CutImg(j,SeamVector(j,k))==0;
                        CutImg(j,SeamVector(j,k))= 2;
                    end
                end
            end
        end
    end
    x=CutImg;
    clear CutImg
    [rows cols]=size(x);
end