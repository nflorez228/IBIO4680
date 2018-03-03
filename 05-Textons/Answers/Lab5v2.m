addpath('lib/matlab')
tic
%%Create a filter bank with deafult params
[fb] = fbCreate;
A=dir('train');
imBase=[];
vect={};
for j=3:numel(A)
D=dir(['train/' A(j).name '/*.jpg']);

for i=1:numel(D)
    D(i).name
    imtemp = double(imread(D(i).name))/255;
    imtemp = imtemp(1:50,1:50);
    imBase=[imBase imtemp];
    vect{end+1}=A(j).name;

end
end
k = 16*8;

%Apply filterbank to sample image
filterResponses=fbRun(fb,imBase)

%Computer textons from filter
[map,textons] = computeTextons(filterResponses,k);
histograma=[];
A=dir('train');
for j=3:numel(A)
    D=dir(['train/' A(j).name '/*.jpg']);
    for i=1:numel(D)
        tmapBase1 = assignTextons(fbRun(fb,double(imread(D(i).name))/255),textons');
        histograma=[histograma histc(tmapBase1(:),1:k)/numel(tmapBase1)]
    end
end
toc
save('hsitograma_train50.mat')
