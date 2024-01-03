function [imgout,background] = BKGsub_rollingball(img,radius,lightbkg,isparaboloid,ispresmooth)
%BKGSUB_ROLLINGBALL Summary of this function goes here
%   Detailed explanation goes here
% img:              input image
% radius:           The radius of the rolling ball
% lightbkg:
% isparaboloid:
% ispresmooth:      if smooth the img before processing?
%% format parameters
if nargin<2
    error("No enough arguments. The function should have at least two arguments (img, radius).")
elseif nargin<3
    lightbkg = false;
    isparaboloid = false;
    ispresmooth = false;
elseif nargin<4
    isparaboloid = false;
    ispresmooth = false;
elseif nargin<5
    ispresmooth = false;
end
imgclass = string(class(img));
imgdouble = double(img); % Set the image data type as "double"

%% Use presmooth?
if ispresmooth
    filterwindow = 3;
    imgdoubleout = imfilter(imgdouble,ones(filterwindow)/(filterwindow^2),'same');
else
    imgdoubleout = imgdouble;
end
%% Use paraboloid?
if ~isparaboloid
    [balldata,ballwidth,ballshrinkfactor] = rolling_ball(radius);
    ball = {balldata,ballwidth,ballshrinkfactor};
    imgdoubleout = BKGsub_rollingball_core(imgdoubleout,imgclass,ball,lightbkg);
else
    imgdoubleout = BKGsub_slidingparaboloid_core(imgdoubleout,imgclass,radius,lightbkg);
end
background = imgdoubleout;
imagedata = imgdouble-background;
imagedata(imagedata<0)=0;
imgout = imagedata;

end