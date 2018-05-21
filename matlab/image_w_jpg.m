function image_w_jpg(X,file_name,image,qfflag,qf)

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