function [map,textons] = computeTextons(fim,k)
% function [map,textons] = computeTextons(fim,k)

%%Load sample images from disk
%imBase1=double(rgb2gray(imread('~/Lab05-Textons/img/person1.bmp')))/255;
%imBase2=double(rgb2gray(imread('~/Lab05-Textons/img/goat1.bmp')))/255;

%Set number of clusters
%k = 16*8;

%Apply filterbank to sample image
%filterResponses=fbRun(fb,horzcat(imBase1,imBase2))

%Computer textons from filter
%[map,textons] = computeTextons(filterResponses,k);

%Load more images
%imTest1=double(rgb2gray(imread('/home/fuanka/Vision17/Lab5-features/img/person2.bmp')))/255;
%imTest2=double(rgb2gray(imread('/home/fuanka/Vision17/Lab5-features/img/goat2.bmp')))/255;

d = numel(fim);
n = numel(fim{1});
data = zeros(d,n);
for i = 1:d,
  data(i,:) = fim{i}(:)';
end

[map,textons] = kmeans(data',k);
[w,h] = size(fim{1});
map = reshape(map,w,h);


