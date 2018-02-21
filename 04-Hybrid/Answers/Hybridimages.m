clear all
close all
clc
m_FirstImg = imread('gordisc.jpg');
m_SecondImg = imread('1.jpeg');
imshow(m_FirstImg)
figure
imshow(m_SecondImg)
figure

m_FilterFirstImg = [1,1,1;1,1,1;1,1,1]/9;
m_FilterSecondImg = [1,1,1;1,1,1;1,1,1]/9;


m_FilteredFirstImg = imfilter(m_FirstImg, m_FilterFirstImg);
m_FilteredSecondImg = imfilter(m_SecondImg, m_FilterSecondImg);

for i=1:50
m_FilteredFirstImg = imfilter(m_FilteredFirstImg, m_FilterFirstImg);
m_FilteredSecondImg = imfilter(m_FilteredSecondImg, m_FilterSecondImg);
    
end

imshow(m_FilteredFirstImg)
figure
imshow(m_FilteredSecondImg)
figure

m_FilteredSecondImg = m_SecondImg - m_FilteredSecondImg;
imshow(m_FilteredSecondImg)
figure

m_ResultingImg = m_FilteredFirstImg + m_FilteredSecondImg;

imshow(m_ResultingImg)

%%

clear all
close all
clc
m_FirstImg = imread('gordisc.jpg');
m_SecondImg = imread('1.jpeg');
imshow(m_FirstImg)
figure
imshow(m_SecondImg)
figure
m_MergedImage=[m_FirstImg(:,1:end/2,:) m_SecondImg(:,end/2+1:end,:)];
imshow(m_MergedImage)
figure
m_Filter = fspecial('gaussian');

    m_result=m_MergedImage;
for i=1:4

m_FilteredImg = imfilter(m_result, m_Filter);
for i=1:50
    m_FilteredImg = imfilter(m_FilteredImg, m_Filter);
end
imshow(m_FilteredImg)
figure

m_downSampled = imresize(m_FilteredImg,0.5);
m_upSampled = imresize(m_downSampled,2);

imshow(m_downSampled)
figure
imshow(m_upSampled)
figure
m_result = m_MergedImage - (m_MergedImage - m_upSampled);
imshow(m_result)

end

