function seamcarving_h(X,image,file_name_p,file2_name_p,file4_name_p,qf,quality_factor,seam_remove,qfflag,src,markflag,premark,sc_cu)

if markflag ==1
    X=double(X)/255;
end
[rows cols dim]=size(X);
Y=rgb2hsv(X);
E=findEnergy(X);
if markflag ==1
    mark = zeros(rows,cols);
else
    mark=premark;
end
seam_remove_l = [5 10 20 30 40 50];
%fprintf(' size(mark) befor is [%d %d] \n',size(mark));
src_l =6;
sr=1;
t1=rows;


srr =min(int64(t1*(seam_remove(sr)/100)),rows-1);
srr2 =t1*(seam_remove(sr)/100);
i=1 ;
while(i<= srr)
    S=findSeamImg_h(E);
    %find seam vector given input "energy map" seam calculation image
    SeamVector=findSeam_h(S);
    %remove seam from image
    X = SeamCut_h(X,SeamVector);
    E = SeamCut_h(E,SeamVector);
    mark = markcut_h(mark,SeamVector,markflag);
    

    %updates size of image
    [rows cols dim]=size(X);
    
    if i== srr
            %% write image
        if qfflag==1
            for qff =1:4
                if markflag ==1
                    file_name = [file_name_p,num2str(quality_factor(qff)),'\seamcarving',num2str(seam_remove(sr)),'_h\'];
                    file2_name =[file2_name_p,num2str(quality_factor(qff)),'\seamcarving',num2str(seam_remove(sr)),'_h_txt\'];
                    file4_name =[file4_name_p,num2str(seam_remove(sr)),'_h\'];
                else
                    file_name = [file_name_p,num2str(quality_factor(qff)),'\seamcarving',num2str(sc_cu),'_l_noise\','\seamcarving',num2str(seam_remove(sr)),'_h\'];
                    file2_name =[file2_name_p,num2str(quality_factor(qff)),'\seamcarving',num2str(sc_cu),'_l_noise\','\seamcarving',num2str(seam_remove(sr)),'_h_txt\'];
                    file4_name =[file4_name_p,num2str(seam_remove(sr)),'_h\'];
                end
                if length(num2str(image))==1
                    file3_name = [file2_name,'ucid0000',num2str(image),'.txt'];
                elseif length(num2str(image))==2
                    file3_name = [file2_name,'ucid000',num2str(image),'.txt'];
                elseif length(num2str(image))==3
                    file3_name = [file2_name,'ucid00',num2str(image),'.txt'];
                elseif length(num2str(image))==4
                    file3_name = [file2_name,'ucid0',num2str(image),'.txt'];
                end
                image_w_jpg(X,file_name,image,1,quality_factor(qff));
                fid = fopen(file3_name, 'wt' );
                for ij = 1:rows
                    for ii = 1:cols
                        fprintf(fid,'%d',mark(ij,ii));
                    end
                    fprintf(fid,'\n');
                end
                fclose(fid);
            end
            %pure
            image_w(X,file4_name,image);
        else
            if markflag ==1
                file_name = [file_name_p,num2str(qf),'\seamcarving',num2str(seam_remove(sr)),'_h\'];
                file2_name =[file2_name_p,num2str(qf),'\seamcarving',num2str(seam_remove(sr)),'_h_txt\'];
            else
                file_name = [file_name_p,num2str(qf),'\seamcarving',num2str(sc_cu),'_l_noise\','\seamcarving',num2str(seam_remove(sr)),'_h\'];
                file2_name =[file2_name_p,num2str(qf),'\seamcarving',num2str(sc_cu),'_l_noise\','\seamcarving',num2str(seam_remove(sr)),'_h_txt\'];
            end
            if length(num2str(image))==1
                file3_name = [file2_name,'ucid0000',num2str(image),'.txt'];
            elseif length(num2str(image))==2
                file3_name = [file2_name,'ucid000',num2str(image),'.txt'];
            elseif length(num2str(image))==3
                file3_name = [file2_name,'ucid00',num2str(image),'.txt'];
            elseif length(num2str(image))==4
                file3_name = [file2_name,'ucid0',num2str(image),'.txt'];
            end
            image_w_jpg(X,file_name,image,0,0);
            fid = fopen(file3_name, 'wt' );
            for ij = 1:rows
                for ii = 1:cols
                    fprintf(fid,'%d',mark(ij,ii));
                end
                fprintf(fid,'\n');
            end
            fclose(fid);
        end
          %% write txt file(pure)
        file5_name =[file4_name_p,num2str(seam_remove(sr)),'_h_txt\'];
        if length(num2str(image))==1
            file6_name = [file5_name,'ucid0000',num2str(image),'.txt'];
        elseif length(num2str(image))==2
            file6_name = [file5_name,'ucid000',num2str(image),'.txt'];
        elseif length(num2str(image))==3
            file6_name = [file5_name,'ucid00',num2str(image),'.txt'];
        elseif length(num2str(image))==4
            file6_name = [file5_name,'ucid0',num2str(image),'.txt'];
        end
        if qfflag ==1
            fid = fopen(file6_name, 'wt' );
            for ij = 1:rows
                for ii = 1:cols
                    fprintf(fid,'%d',mark(ij,ii));
                end
                fprintf(fid,'\n');
            end
            fclose(fid);
        end
        %% noise
        if markflag ==1
            file_l_name = [file_name_p];
            file2_l_name = [file2_name_p];
            file4_l_name = [file4_name_p,num2str(seam_remove(sr)),'_h_noise\'];
           for qff =1:4
                dos(['mkdir ', file_name_p,num2str(quality_factor(qff)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\']);
                dos(['mkdir ', file4_name_p,num2str(seam_remove(sr)),'_h_noise\']);
                for sr_h = 1:src_l
                    dos(['mkdir ', file_l_name,num2str(quality_factor(qff)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\','seamcarving',num2str(seam_remove_l(sr_h)),'_l\']);
                    dos(['mkdir ', file2_l_name,num2str(quality_factor(qff)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\','seamcarving',num2str(seam_remove_l(sr_h)),'_l_txt\']);
                    dos(['mkdir ', file4_l_name,'\seamcarving',num2str(seam_remove_l(sr_h)),'_l\']);
                    dos(['mkdir ', file4_l_name,'\seamcarving',num2str(seam_remove_l(sr_h)),'_l_txt\']);
                end
            end
            file_l_name = [file_name_p];
            file2_l_name = [file_name_p];
            file4_l_name = [file4_name_p,num2str(seam_remove(sr)),'_h_noise\','\seamcarving'];
            seamcarving(X,image,file_l_name,file2_l_name,file4_l_name,qf,quality_factor,seam_remove_l,qfflag,src_l,0,mark,seam_remove(sr));
        end
        %% control
        k=srr;
        if sr < src
            sr =sr+1;
            srr =min(int64(t1*(seam_remove(sr)/100)),rows-1);
            srr2 =t1*(seam_remove(sr)/100);
            if srr-int64(k)<0
                i=i-1;
            end
        end
        
    end
    i =i+1;
end

%fprintf(' size(mark) after is [%d %d] \n',size(mark));

