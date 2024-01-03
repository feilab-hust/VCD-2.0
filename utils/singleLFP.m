function [center_LF] = singleLFP(img,H)

    global zeroImageEx;
    global exsize;
    [h,w,~]=size(img);
    xsize = [h, w];
    msize = [size(H,1), size(H,2)];
    mmid = floor(msize/2);
    exsize = xsize + mmid;
    exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];
    zeroImageEx = gpuArray(zeros(exsize, 'single'));
    center_LF=gather(shift_forward_projection(H,img));

end




