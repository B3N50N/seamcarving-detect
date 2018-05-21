close all
clear 
clc

mark = zeros(1,1);
percent = [10 20 30 50];
quality_factor = [25 50 75 100];
seam_remove = [10 20 30 40 50];
%seam_remove = [11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29 31 32 33 34 35 36 37 38 39 41 42 43 44 45 46 47 48 49];
noise = [5 10 20 30 40 50];
smooth =[3 5 7];
sharp_Radius = [1 1.5];
sharp_Amount = [0.6 1.8];%0.6 1.2 1.8
sharp_Threshold = [0 0.7];% 0 0.4 0.7
src = 5;%36
noisec = 6;
smoothc = 3;
sharp_Radiusc =  2;
sharp_Amountc =  2;
sharp_Thresholdc = 2;
work_dir = 'C:\Users\¿à±á³ó\Desktop\¾Ç³ø°Q½×\sampleimage\';
%work_dir = [pwd,'\'];
image_c=30;%1338
for qf = 1:4
    dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\source_image']);
    dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\source_image']);
    dos(['mkdir ', work_dir,'pure\QF_pure\source_image']);
    for sr = 1:src
    dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l\']);
    dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_txt\']);
    dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h\']);
    dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_txt\']);
    dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l\']);
    dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_txt\']);
    dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h\']);
    dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_txt\']);
    dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l\']);
    dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_txt\']);
    dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h\']);
    dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_txt\']);
    end
    for image = 1:image_c
        RGB=image_r([work_dir,'source_image\'],image);
        file_name = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\source_image\'];
        image_w_jpg(RGB,file_name,image,1,quality_factor(qf));
        file_name = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\source_image\'];
        image_w_jpg(RGB,file_name,image,1,quality_factor(qf));
    end
end
%% pre
parfor image = 1:image_c
    RGB=image_r([work_dir,'source_image\'],image);
    X = RGB;
    file_name = [work_dir,'Pre_QF\QF_'];
    file2_name =[work_dir,'Pre_QF\QF_'];
    file4_name =[work_dir,'pure\QF_pure\seamcarving'];
    seamcarving(X,image,file_name,file2_name,file4_name,quality_factor(qf),quality_factor,seam_remove,1,src,1,mark,0);

    X = RGB;
    file_name = [work_dir,'Pre_QF\QF_'];
    file2_name =[work_dir,'Pre_QF\QF_'];
    file4_name =[work_dir,'pure\QF_pure\seamcarving'];
    seamcarving_h(X,image,file_name,file2_name,file4_name,quality_factor(qf),quality_factor,seam_remove,1,src,1,mark,0);
end
%% pos
for qf = 1:4
    parfor image = 1:image_c
        
        file_name = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\source_image\'];
        RGB = image_r_jpg(file_name,image);
        X = RGB;
        file_name = [work_dir,'Pos_QF\QF_'];
        file2_name =[work_dir,'Pos_QF\QF_'];
        file4_name =[work_dir,'pure\QF_pure\seamcarving'];
        seamcarving(X,image,file_name,file2_name,file4_name,quality_factor(qf),quality_factor,seam_remove,0,src,1,mark,0);
        
        X = RGB;
        file_name = [work_dir,'Pos_QF\QF_'];
        file2_name =[work_dir,'Pos_QF\QF_'];
        file4_name =[work_dir,'pure\QF_pure\seamcarving'];
        seamcarving_h(X,image,file_name,file2_name,file4_name,quality_factor(qf),quality_factor,seam_remove,0,src,1,mark,0);
    end
end
%% smooth
for sm = 1:smoothc
    for sr = 1:src
        dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_smooth_',num2str(smooth(sm))]);
        dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_smooth_',num2str(smooth(sm))]);
        for no = 1:noisec
            dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_smooth_',num2str(smooth(sm))]);
            dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_smooth_',num2str(smooth(sm))]);
        
        end
    end
end
for qf = 1:4
    for sm = 1:smoothc
        for sr = 1:src
            dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_smooth_',num2str(smooth(sm))]);
            dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_smooth_',num2str(smooth(sm))]);
            dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_smooth_',num2str(smooth(sm))]);
            dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_smooth_',num2str(smooth(sm))]);
            for no = 1:noisec
                dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_smooth_',num2str(smooth(sm))]);
                dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_smooth_',num2str(smooth(sm))]);
                dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_smooth_',num2str(smooth(sm))]);
                dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_smooth_',num2str(smooth(sm))]);

            end
        end
    end
end
for sm = 1:smoothc
    parfor image = 1:image_c
        for sr = 1:src
            file_name = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l\'];
            file_name2 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h\'];
            file_name3 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_smooth_',num2str(smooth(sm)),'\'];
            file_name4 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_smooth_',num2str(smooth(sm)),'\'];
                        
            RGB = image_r(file_name,image);
            X = imgaussfilt(RGB,smooth(sm));
            image_w(X,file_name3,image);
            
            RGB = image_r(file_name2,image);
            X = imgaussfilt(RGB,smooth(sm));
            image_w(X,file_name4,image);
            
            for no = 1:noisec
                file_name5 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h\'];
                file_name6 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l\'];
                file_name7 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_smooth_',num2str(smooth(sm)),'\'];
                file_name8 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_smooth_',num2str(smooth(sm)),'\'];
                
                RGB = image_r(file_name5,image);
                X = imgaussfilt(RGB,smooth(sm));
                image_w(X,file_name7,image);

                RGB = image_r(file_name6,image);
                X = imgaussfilt(RGB,smooth(sm));
                image_w(X,file_name8,image);
            end
        end
    end
end
for qf = 1:4
    for sm = 1:smoothc
        parfor image = 1:image_c
            for sr = 1:src
                file_name = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l\'];
                file_name2 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h\'];
                file_name3 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_smooth_',num2str(smooth(sm)),'\'];
                file_name4 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_smooth_',num2str(smooth(sm)),'\'];
                file_name5 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l\'];
                file_name6 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h\'];
                file_name7 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_smooth_',num2str(smooth(sm)),'\'];
                file_name8 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_smooth_',num2str(smooth(sm)),'\'];

                RGB = image_r_jpg(file_name,image);
                X = imgaussfilt(RGB,smooth(sm));
                image_w_jpg(X,file_name3,image,0,0);

                RGB = image_r_jpg(file_name2,image);
                X = imgaussfilt(RGB,smooth(sm));
                image_w_jpg(X,file_name4,image,0,0);
                
                RGB = image_r_jpg(file_name5,image);
                X = imgaussfilt(RGB,smooth(sm));
                image_w_jpg(X,file_name7,image,0,0);

                RGB = image_r_jpg(file_name6,image);
                X = imgaussfilt(RGB,smooth(sm));
                image_w_jpg(X,file_name8,image,0,0);

                for no = 1:noisec
                    file_name = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h\'];
                    file_name2 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l\'];
                    file_name3 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_smooth_',num2str(smooth(sm)),'\'];
                    file_name4 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_smooth_',num2str(smooth(sm)),'\'];
                    file_name5 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h\'];
                    file_name6 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l\'];
                    file_name7 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_smooth_',num2str(smooth(sm)),'\'];
                    file_name8 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_smooth_',num2str(smooth(sm)),'\'];

                    RGB = image_r_jpg(file_name,image);
                    X = imgaussfilt(RGB,smooth(sm));
                    image_w_jpg(X,file_name3,image,0,0);

                    RGB = image_r_jpg(file_name2,image);
                    X = imgaussfilt(RGB,smooth(sm));
                    image_w_jpg(X,file_name4,image,0,0);
                    
                    RGB = image_r_jpg(file_name5,image);
                    X = imgaussfilt(RGB,smooth(sm));
                    image_w_jpg(X,file_name7,image,0,0);

                    RGB = image_r_jpg(file_name6,image);
                    X = imgaussfilt(RGB,smooth(sm));
                    image_w_jpg(X,file_name8,image,0,0);
                end
            end
        end
    end
end
%% sharpen
for spr = 1:sharp_Radiusc
    for spa = 1:sharp_Amountc
        for spt = 1:sharp_Thresholdc
            for sr = 1:src
                dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                for no = 1:noisec
                    dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                    dos(['mkdir ', work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);

                end
            end
        end
    end
end
for qf = 1:4
    for spr = 1:sharp_Radiusc
        for spa = 1:sharp_Amountc
            for spt = 1:sharp_Thresholdc
                for sr = 1:src
                    dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                    dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                    dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                    dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                    for no = 1:noisec
                        dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                        dos(['mkdir ', work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                        dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);
                        dos(['mkdir ', work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt))]);

                    end
                end
            end
        end
    end
end
for spr = 1:sharp_Radiusc
    for spa = 1:sharp_Amountc
        for spt = 1:sharp_Thresholdc
            parfor image = 1:image_c
                for sr = 1:src
                    file_name = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l\'];
                    file_name2 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h\'];
                    file_name3 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];
                    file_name4 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];

                    RGB = image_r(file_name,image);
                    X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                    image_w(X,file_name3,image);

                    RGB = image_r(file_name2,image);
                    X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                    image_w(X,file_name4,image);

                    for no = 1:noisec
                        file_name5 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h\'];
                        file_name6 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l\'];
                        file_name7 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];
                        file_name8 = [work_dir,'pure\QF_pure\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];

                        RGB = image_r(file_name5,image);
                        X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                        image_w(X,file_name7,image);

                        RGB = image_r(file_name6,image);
                        X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                        image_w(X,file_name8,image);
                    end
                end
            end
        end
    end
end
for qf = 1:4
    for spr = 1:sharp_Radiusc
        for spa = 1:sharp_Amountc
            for spt = 1:sharp_Thresholdc
                parfor image = 1:image_c
                    for sr = 1:src
                        file_name = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l\'];
                        file_name2 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h\'];
                        file_name3 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];
                        file_name4 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];
                        file_name5 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l\'];
                        file_name6 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h\'];
                        file_name7 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];
                        file_name8 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];

                        RGB = image_r_jpg(file_name,image);
                        X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                        image_w_jpg(X,file_name3,image,0,0);

                        RGB = image_r_jpg(file_name2,image);
                        X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                        image_w_jpg(X,file_name4,image,0,0);

                        RGB = image_r_jpg(file_name5,image);
                        X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                        image_w_jpg(X,file_name7,image,0,0);

                        RGB = image_r_jpg(file_name6,image);
                        X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                        image_w_jpg(X,file_name8,image,0,0);

                        for no = 1:noisec
                            file_name = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h\'];
                            file_name2 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l\'];
                            file_name3 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];
                            file_name4 = [work_dir,'Pre_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];
                            file_name5 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h\'];
                            file_name6 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l\'];
                            file_name7 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_l_noise\seamcarving',num2str(noise(no)),'_h_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];
                            file_name8 = [work_dir,'Pos_QF\QF_',num2str(quality_factor(qf)),'\seamcarving',num2str(seam_remove(sr)),'_h_noise\seamcarving',num2str(noise(no)),'_l_sharp_',num2str(sharp_Radius(spr)),'_',num2str(sharp_Amount(spa)),'_',num2str(sharp_Threshold(spt)),'\'];

                            RGB = image_r_jpg(file_name,image);
                            X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                            image_w_jpg(X,file_name3,image,0,0);

                            RGB = image_r_jpg(file_name2,image);
                            X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                            image_w_jpg(X,file_name4,image,0,0);

                            RGB = image_r_jpg(file_name5,image);
                            X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                            image_w_jpg(X,file_name7,image,0,0);

                            RGB = image_r_jpg(file_name6,image);
                            X = imsharpen(RGB,'Radius',sharp_Radius(spr),'Amount',sharp_Amount(spa),'Threshold',sharp_Threshold(spt));
                            image_w_jpg(X,file_name8,image,0,0);
                        end
                    end
                end
            end
        end
    end
end