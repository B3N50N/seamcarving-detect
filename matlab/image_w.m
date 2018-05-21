function image_w(X,file_name,image)

if length(num2str(image))==1
    imwrite(X,[file_name,'ucid0000',num2str(image),'.tif']);
elseif length(num2str(image))==2
    imwrite(X,[file_name,'ucid000',num2str(image),'.tif']);
elseif length(num2str(image))==3
    imwrite(X,[file_name,'ucid00',num2str(image),'.tif']);    
else
    imwrite(X,[file_name,'ucid0',num2str(image),'.tif']);
end