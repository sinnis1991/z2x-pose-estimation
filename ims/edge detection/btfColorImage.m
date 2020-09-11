
function J=btfColorImage(I,G,sigma_d,sigma_r,filterRadius)
x=-filterRadius:filterRadius;
y=-filterRadius:filterRadius;
[xx,yy]=meshgrid(x,y);
spatialKernel=exp(- (xx.^2+yy.^2)/(2*sigma_d^2));
[rows,cols,channels]=size(I);
rc=zeros(size(I(:,:,1)));
gc=zeros(size(rc));
bc=zeros(size(gc));
if size(G,3)==1
    temp=G;
    G(:,:,1)=temp;
    G(:,:,2)=temp;
    G(:,:,3)=temp;
end
 
parfor y=filterRadius+1:rows-filterRadius
    for x=filterRadius+1:cols-filterRadius
        roi= I(y-filterRadius:y+filterRadius,x-filterRadius:x+filterRadius,:);
        roidif=zeros(size(roi));
        roidif(:,:,1)=roi(:,:,1)-G(y,x,1);
        roidif(:,:,2)=roi(:,:,2)-G(y,x,2);
        roidif(:,:,3)=roi(:,:,3)-G(y,x,3);
        roidif=roidif.^2;
        roidif=roidif(:,:,1)+roidif(:,:,2)+roidif(:,:,3);
        tonalKernel =exp(- roidif/(2*sigma_r^2));
        W=(tonalKernel.*spatialKernel);
        k=sum(W(:));
        RC=W.*roi(:,:,1);
        GC=W.*roi(:,:,2);
        BC=W.*roi(:,:,3);
        rc(y,x)=sum(RC(:))/k;
        gc(y,x)=sum(GC(:))/k;
        bc(y,x)=sum(BC(:))/k;
    end
end
rc=rc(filterRadius+1:end-filterRadius,filterRadius+1:end-filterRadius);
gc=gc(filterRadius+1:end-filterRadius,filterRadius+1:end-filterRadius);
bc=bc(filterRadius+1:end-filterRadius,filterRadius+1:end-filterRadius);
J=cat(3,rc,gc,bc);



