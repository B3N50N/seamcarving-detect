function RGB = image_r(file_name,image)
if length(num2str(image))==1
    RGB = imread([file_name,'ucid0000',num2str(image),'.tif']);
elseif length(num2str(image))==2
    RGB = imread([file_name,'ucid000',num2str(image),'.tif']);
elseif length(num2str(image))==3
    RGB = imread([file_name,'ucid00',num2str(image),'.tif']);    
else
    RGB = imread([file_name,'ucid0',num2str(image),'.tif']);
end