function RGB = image_r_jpg(file_name,image)
if length(num2str(image))==1
    RGB = imread([file_name,'ucid0000',num2str(image),'.jpg']);
elseif length(num2str(image))==2
    RGB = imread([file_name,'ucid000',num2str(image),'.jpg']);
elseif length(num2str(image))==3
    RGB = imread([file_name,'ucid00',num2str(image),'.jpg']);    
else
    RGB = imread([file_name,'ucid0',num2str(image),'.jpg']);
end