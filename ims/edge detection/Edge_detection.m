clear all;
% g = double(imread('path-to-your-image.png'))/255.0;

g = double(imread('../experiment/1/a_.png'))/255.0;
g = cat(3, g, g, g);

G=g;%guidance image
sigma_d=2;
sigma_r=0.1;
filterSize=double(uint8(3*sigma_d)*2+1);
 
filterRadius=ceil((filterSize-1)/2);
I=padarray(g,[filterRadius,filterRadius],'replicate');
G=padarray(G,[filterRadius,filterRadius],'replicate');
 
J=btfColorImage(I,G,sigma_d,sigma_r,filterRadius);

I = rgb2gray(J);

I2 = I(:,1:480);
I2 = imresize(I2,[128*2,128*2]);
BW1 = edge(I2,'Canny',0.0705882,0.2);
se = strel('cube',3);
I2 = imdilate(BW1,se);
I2 = imresize(I2,[128,128]);
imwrite(I2,'Edge_L.png')

I2 = I(:,81:560);
I2 = imresize(I2,[128*2,128*2]);
BW1 = edge(I2,'Canny',0.0705882,0.2);
se = strel('cube',3);
I2 = imdilate(BW1,se);
I2 = imresize(I2,[128,128]);
imwrite(I2,'Edge_M.png')

I2 = I(:,161:end);
I2 = imresize(I2,[128*2,128*2]);
BW1 = edge(I2,'Canny',0.0705882,0.2);
se = strel('cube',3);
I2 = imdilate(BW1,se);
I2 = imresize(I2,[128,128]);
imwrite(I2,'Edge_R.png')



