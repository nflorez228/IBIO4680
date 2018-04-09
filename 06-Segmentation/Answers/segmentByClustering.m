function segmentation = segmentByClustering( rgbImage, featureSpace, clusteringMethod, numberOfClusters)

if nargin < 4
    error('Bad number of entry parameters')
end

if numberOfClusters < 2
    error('Bad number of clusters')
end

%he = imread(rgbImage);
he=rgbImage;
imshow(he);
he=im2double(he);
validForWatersheds=1;
switch featureSpace
    case 'rgb'
        %nothing happens as the image is already in rgb
    case 'lab'
        he=rgb2lab(he);
    case 'hsv'
        he=rgb2hsv(he);
    case 'rgb+xy'
        validForWatersheds=0;
        for i=1:size(rgbImage,1)
            for j=1:size(rgbImage,2)
                he(i,j,4)=j;
                he(i,j,5)=i;
            end
        end
    case 'lab+xy'
        validForWatersheds=0;
        he=rgb2lab(he);
        for i=1:size(rgbImage,1)
            for j=1:size(rgbImage,2)
                he(i,j,4)=j;
                he(i,j,5)=i;
            end
        end
    case 'hsv+xy'
        validForWatersheds=0;
        he=rgb2hsv(he);
        for i=1:size(rgbImage,1)
            for j=1:size(rgbImage,2)
                he(i,j,4)=j;
                he(i,j,5)=i;
            end
        end
    otherwise
        error('Incorrect color space, it must be: rgb, lab, hsv, rgb+xy, lab+xy, hsv+xy')
end
rows=size(he,1);
cols=size(he,2);
switch clusteringMethod
    case 'kmeans'
        l=0;
        for i=1:rows
            for j=1:cols
                l=l+1;
                X(l,:)=he(i,j,:);
            end
        end
        [cluster_idx, cluster_center] = kmeans(X,numberOfClusters);
        idxpos=1;
        for i=1:rows
            for j=1:cols
                segs(i,j)=cluster_idx(idxpos);
                idxpos=idxpos+1;
            end
        end
        image(segs)
        colormap colorcube
    case 'gmm'
        l=0;
        for i=1:rows
            for j=1:cols
                l=l+1;
                X(l,:)=he(i,j,:);
            end
        end
        obj=fitgmdist(X,numberOfClusters);
        idx=cluster(obj,X);
        idxpos=1;
        for i=1:rows
            for j=1:cols
                segs(i,j)=idx(idxpos);
                idxpos=idxpos+1;
            end
        end
        image(segs)
        colormap colorcube
        
    case 'hierarchical'
        he=imresize(he,0.1);
        rows=size(he,1);
        cols=size(he,2);
        l=0;
        for i=1:rows
            for j=1:cols
                l=l+1;
                X(l,:)=he(i,j,:);
            end
        end       
        Z = linkage(X);
        cluster_idx = cluster(Z,'maxclust',numberOfClusters);
        idxpos=1;
        for i=1:rows
            for j=1:cols
                segs(i,j)=cluster_idx(idxpos);
                idxpos=idxpos+1;
            end
        end
        image(segs)
        colormap colorcube
    case 'watershed'
        if validForWatersheds
            he=im2uint8(he);
            he=rgb2gray(he);
            hy= fspecial('sobel');
            hx = hy';
            Iy = imfilter(double(he), hy, 'replicate');
            Ix = imfilter(double(he), hx, 'replicate');
            gradmag = sqrt(Ix.^2 + Iy.^2);
            figure
            imshow(gradmag,[]), title('Gradient magnitude (gradmag)')
            marker = imextendedmin(gradmag,numberOfClusters);
            new_grad = imimposemin(gradmag,marker);
            ws = watershed(new_grad);
            figure;
            imshow(ws == 0)
            figure
            segs=label2rgb(ws, 'colorcube', 'w', 'shuffle');
            imshow(label2rgb(ws, 'colorcube', 'w', 'shuffle'))
        else
            error('Spaces with +xy are not valid for watershed, use rgb, lab, hsv instead.')
        end
    otherwise
        error('Incorrect clustering method, it must be: kmeans, gmm, hierarchical, watershed')
end
segmentation =segs;
end

