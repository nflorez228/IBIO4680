load('data2/deteccion_caras.mat');

prediccion = {};
prediccion{1,1}= '0_Parade_Parade_0_960';
prediccion{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end

load('data2/deteccion_caras2.mat');
d=0;
prediccion2{1,1}= '61_Street_Battle_streetfight_61_936';
prediccion2{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion2{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end

load('data2/deteccion_caras3.mat');

d=0;
prediccion3{1,1}= '9_Press_Conference_Press_Conference_9_945';
prediccion3{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion3{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end

load('data2/deteccion_caras4.mat');

d=0;
prediccion4{1,1}= '8_Election_Campain_Election_Campaign_8_620';
prediccion4{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion4{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end

load('data2/deteccion_caras5.mat');

d=0;
prediccion5{1,1}= '7_Cheering_Cheering_7_884';
prediccion5{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion5{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end
load('data2/deteccion_caras6.mat');

d=0;
prediccion6{1,1}= '6_Funeral_Funeral_6_1029';
prediccion6{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion6{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end

load('data2/deteccion_caras7.mat');
d=0;
prediccion7{1,1}= '5_Car_Accident_Car_Crash_5_868';
prediccion7{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion7{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end

load('data2/deteccion_caras8.mat');

d=0;
prediccion8{1,1}= '4_Dancing_Dancing_4_1043';
prediccion8{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion8{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end

load('data2/deteccion_caras9.mat');

d=0;
prediccion9{1,1}= '3_Riot_Riot_3_1037';
prediccion9{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion9{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end

load('data2/deteccion_caras10.mat');

d=0;
prediccion10{1,1}= '2_Demonstration_Protesters_2_1033';
prediccion10{2,1}= [length(scores)];

for d = 3:1:length(scores)+2
    width = detections(3,d-2)-detections(1,d-2);
    Higth = detections(4,d-2)-detections(2,d-2);
    prediccion10{d,1} = [detections(3,d-2) detections(4,d-2) width Higth scores(1,d-2)];
end


fileID = fopen('prediccion.txt','w');

formatSpec = '%s\n';
formatSpec2 = '%d\n';
formatSpec3 = '%2.4f %2.4f %2.4f %2.4f %2.4f\n';

[nrows,ncols] = size(prediccion);

fprintf(fileID,formatSpec,prediccion{1,1});
fprintf(fileID,formatSpec2,prediccion{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion{row,:});
end

[nrows,ncols] = size(prediccion2);

fprintf(fileID,formatSpec,prediccion2{1,1});
fprintf(fileID,formatSpec2,prediccion2{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion2{row,:});
end

[nrows,ncols] = size(prediccion3);

fprintf(fileID,formatSpec,prediccion3{1,1});
fprintf(fileID,formatSpec2,prediccion3{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion3{row,:});
end

[nrows,ncols] = size(prediccion4);

fprintf(fileID,formatSpec,prediccion4{1,1});
fprintf(fileID,formatSpec2,prediccion4{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion4{row,:});
end

[nrows,ncols] = size(prediccion5);

fprintf(fileID,formatSpec,prediccion5{1,1});
fprintf(fileID,formatSpec2,prediccion5{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion5{row,:});
end

[nrows,ncols] = size(prediccion6);

fprintf(fileID,formatSpec,prediccion6{1,1});
fprintf(fileID,formatSpec2,prediccion6{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion6{row,:});
end

[nrows,ncols] = size(prediccion7);

fprintf(fileID,formatSpec,prediccion7{1,1});
fprintf(fileID,formatSpec2,prediccion7{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion7{row,:});
end

[nrows,ncols] = size(prediccion8);

fprintf(fileID,formatSpec,prediccion8{1,1});
fprintf(fileID,formatSpec2,prediccion8{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion8{row,:});
end

[nrows,ncols] = size(prediccion9);

fprintf(fileID,formatSpec,prediccion9{1,1});
fprintf(fileID,formatSpec2,prediccion9{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion9{row,:});
end

[nrows,ncols] = size(prediccion10);

fprintf(fileID,formatSpec,prediccion10{1,1});
fprintf(fileID,formatSpec2,prediccion10{2,1});

for row = 3:nrows
    fprintf(fileID,formatSpec3,prediccion10{row,:});
end

fclose(fileID);

type prediccion.txt