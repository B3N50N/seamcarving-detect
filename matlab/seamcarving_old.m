function seamcarving(X,image,file_name_p,file2_name_p,qf,seam_remove,qfflag,src)

X=double(X)/255;
[rows cols dim]=size(X);
Y=rgb2hsv(X);
E=findEnergy(X);
mark = zeros(rows,cols);
%fprintf(' size(mark) befor is [%d %d] \n',size(mark));
sr=1;
t1=cols;
srr =min(int64(t1*(seam_remove(sr)/100)),cols-1);
srr2 =t1*(seam_remove(sr)/100);
i=1 ;
while(i<= srr)
    S=findSeamImg(E);
    %find seam vector given input "energy map" seam calculation image
    SeamVector=findSeam(S);

    %remove seam from image
    X = SeamCut(X,SeamVector);
    E = SeamCut(E,SeamVector);
    mark = markcut(mark,SeamVector);

    %updates size of image
    [rows cols dim]=size(X);
    
    if i== srr
            %% make file name
        file_name = [file_name_p,num2str(seam_remove(sr)),'_l\'];
        file2_name =[file2_name_p,num2str(seam_remove(sr)),'_l_txt\'];
        if length(num2str(image))==1
            file3_name = [file2_name,'ucid0000',num2str(image),'.txt'];
        elseif length(num2str(image))==2
            file3_name = [file2_name,'ucid0000',num2str(image),'.txt'];
        elseif length(num2str(image))==3
            file3_name = [file2_name,'ucid0000',num2str(image),'.txt'];
        elseif length(num2str(image))==4
            file3_name = [file2_name,'ucid0000',num2str(image),'.txt'];
        end

            %% write image
        if qfflag==1
            if length(num2str(image))==1
                imwrite(X,[file_name,'ucid0000',num2str(image),'.jpg'],'quality',qf);
            elseif length(num2str(image))==2
                imwrite(X,[file_name,'ucid000',num2str(image),'.jpg'],'quality',qf);
            elseif length(num2str(image))==3
                imwrite(X,[file_name,'ucid00',num2str(image),'.jpg'],'quality',qf);
            elseif length(num2str(image))==4
                imwrite(X,[file_name,'ucid0',num2str(image),'.jpg'],'quality',qf);
            end
        else
             if length(num2str(image))==1
                imwrite(X,[file_name,'ucid0000',num2str(image),'.jpg']);
            elseif length(num2str(image))==2
                imwrite(X,[file_name,'ucid000',num2str(image),'.jpg']);
            elseif length(num2str(image))==3
                imwrite(X,[file_name,'ucid00',num2str(image),'.jpg']);
            elseif length(num2str(image))==4
                imwrite(X,[file_name,'ucid0',num2str(image),'.jpg']);
            end
        end
            
            %% write txt file
        fid = fopen(file3_name, 'wt' );
        for ij = 1:rows
            for ii = 1:cols
            fprintf(fid,'%d ',mark(ij,ii));
            end
            fprintf(fid,'\n');
        end
        fclose(fid);
        %% control
        k=srr;
        if sr < src
            sr =sr+1;
            srr =min(int64(t1*(seam_remove(sr)/100)),cols-1);
            srr2 =t1*(seam_remove(sr)/100);
            if srr-int64(k)<0
                i=i-1;
            end
        end
        
    end
    i =i+1;
end

%fprintf(' size(mark) after is [%d %d] \n',size(mark));
