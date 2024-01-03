function [sr_LF,center_LF] = scanLFP(img,H,shift_times)

center_point=ceil(shift_times/2);
shift_step=size(H,3)/shift_times;
[ori_h,ori_w,depth]=size(img);
Nnum=size(H,3);

new_h=(1-mod(ori_h/Nnum,2))*Nnum+ori_h;
new_w=(1-mod(ori_w/Nnum,2))*Nnum+ori_w;
pd_wf=zeros(new_h,new_w,depth);
pd_wf(1:ori_h,1:ori_w,:)=img;


img= pd_wf;
[h,w,~]=size(img);

global zeroImageEx;
global exsize;
xsize = [h, w];
msize = [size(H,1), size(H,2)];
mmid = floor(msize/2);
exsize = xsize + mmid;
exsize = [ min( 2^ceil(log2(exsize(1))), 128*ceil(exsize(1)/128) ), min( 2^ceil(log2(exsize(2))), 128*ceil(exsize(2)/128) ) ];
zeroImageEx = gpuArray(zeros(exsize, 'single'));
LFP_list=zeros([h,w,shift_times*shift_times],"double");
% forwardFUN = @(Xguess) forwardProjectGPU( H, Xguess );
for v_idx=1:shift_times
    for u_idx=1:shift_times
        x_shift_value=(u_idx-center_point)*shift_step;
        y_shift_value=(v_idx-center_point)*shift_step;
        z_shift_value=0;
        se=imtranslate(img,[y_shift_value x_shift_value  z_shift_value]);
%         se=se + offset ;
        view_idx=(v_idx-1)*shift_times+u_idx;
%         tic
        LFP_list(:,:,view_idx)=shift_forward_projection(H,se);
%         toc
    end
end

center_LF=LFP_list(:,:,ceil(shift_times*shift_times/2));
center_LF=center_LF(1:ori_h,1:ori_w);
%%
Nnum=size(H,3);
temp_indx=1:shift_times;
index1=reshape(repmat(temp_indx,[shift_times,1]),[1,shift_times*shift_times]);
index2=repmat(temp_indx,[1,shift_times]);

Ny=floor(h/Nnum/2); 
Nx=floor(w/Nnum/2); 
centerY=ceil(h/2); 
centerX=ceil(w/2);


% rectification
sLF=zeros( (2*Ny+1)*Nnum,(2*Nx+1)*Nnum,shift_times,shift_times );
for i=1:shift_times*shift_times      
    sample_view_idx=i;
    sLF_slice= LFP_list(:,:,sample_view_idx);
    sLF(:,:,index2(i),index1(i))=sLF_slice(centerY-Nnum*Ny-fix(Nnum/2):centerY+Nnum*Ny+fix(Nnum/2),centerX-Nnum*Nx-fix(Nnum/2):centerX+Nnum*Nx+fix(Nnum/2));
end

% Pixel realignment
multiWDF=zeros(Nnum,Nnum,size(sLF,1)/Nnum,size(sLF,2)/Nnum,shift_times,shift_times); %% multiplexed phase-space
for i=1:Nnum
    for j=1:Nnum
        for a=1:size(sLF,1)/Nnum
            for b=1:size(sLF,2)/Nnum
                multiWDF(i,j,a,b,:,:)=squeeze(  sLF(  (a-1)*Nnum+i,(b-1)*Nnum+j,:,:  )  );
            end
        end
    end
end
WDF=zeros(  size(sLF,1)/Nnum*shift_times,size(sLF,2)/Nnum*shift_times,Nnum,Nnum  ); % multiplexed phase-space
for a=1:size(sLF,1)/Nnum
    for c=1:shift_times
        x=shift_times*a+1-c;
        for b=1:size(sLF,2)/Nnum
            for d=1:shift_times
                y=shift_times*b+1-d;
%                 fprintf('x:%d // y:%d\n',x,y);
                WDF(x,y,:,:)=squeeze(multiWDF(:,:,a,b,c,d));
            end
        end
    end
end


ViewStack=zeros(  size(sLF,1)/Nnum*shift_times,size(sLF,2)/Nnum*shift_times,Nnum*Nnum  );
for ii=1:Nnum
    for jj=1:Nnum
        view_idxxx=(ii-1)*Nnum+jj;
        ViewStack(:,:,view_idxxx)=WDF(:,:,ii,jj);
    end

end
sr_LF= Stack2LFP(ViewStack,Nnum);
sr_LF=sr_LF(1:ori_h*shift_times,1:ori_w*shift_times);

end

