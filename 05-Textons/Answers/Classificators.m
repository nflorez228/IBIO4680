
tic
Mdl = fitcknn(histogramatrain',labelstrain,'NumNeighbors',3);
predict(Mdl,histogramatest')
toc;

%%
tic
Mdl = TreeBagger(5,histogramatrain',labelstrain,'Method','classification')
predict(Mdl,histogramatest')
toc
