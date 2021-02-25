close all; clear all; clc
load cam1_1.mat; load cam2_1.mat; load cam3_1.mat;

[h,w,rgb,frames]=size(vidFrames1_1);
X1=[]; Y1=[];
for j=1:frames
    img=rgb2gray(vidFrames1_1(:,:,:,j));
    img(:,1:310)=0; 
    img(:,390:end)=0; 
    img(1:200,:)=0;
    imshow(img)
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X1=[X1 Ix]; Y1=[Y1 Iy];
end

[h,w,rgb,frames]=size(vidFrames2_1);
X2=[]; Y2=[];
for j=1:frames
    img=rgb2gray(vidFrames2_1(:,:,:,j));
    img(:,1:245)=0; 
    img(:,350:end)=0;
    img(1:100,:)=0;
    img(370:end,:)=0;
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X2=[X2 Ix]; Y2=[Y2 Iy];
end

[h,w,rgb,frames]=size(vidFrames3_1);
X3=[]; Y3=[];
for j=1:frames
    img=rgb2gray(vidFrames3_1(:,:,:,j));
    img(:,1:250)=0;
    img(:,500:end)=0;
    img(1:230,:)=0; 
    img(340:end,:)=0; 
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X3=[X3 Ix]; Y3=[Y3 Iy];
end

Xmin=min([length(X1) length(X2) length(X3)]);
if length(X1) > Xmin
    X1=X1(1:Xmin);
    Y1=Y1(1:Xmin);
end
if length(X2) > Xmin
    X2=X2(1:Xmin);
    Y2=Y2(1:Xmin);
end
if length(X3) > Xmin
    X3=X3(1:Xmin);
    Y3=Y3(1:Xmin);
end

XY=[X1;Y1;X2;Y2;X3;Y3];

figure(1)
for j=1:6
    subplot(3,2,(j)), plot(XY(j,:))
    xlim([0 226])
    ylim([0 500])
    if j==1 || j==2
        title('Camera 1')
    elseif j==3 || j==4
        title('Camera 2')
    elseif j==5 || j==6
        title('Camera 3')
    end
    
    if mod(j,2)==1 %odd
        ylabel('X')
    elseif mod(j,2)==0 %even
        ylabel('Y')
    end
end

[row,col]=size(XY); 
mn=mean(XY,2);
XY=XY-repmat(mn,1,col); 

CovMat=(1/(col-1))*XY*XY'; 
[eVec,eVal]=eig(CovMat); 
lambda=diag(eVal); 
[B,ind]=sort(-1*lambda); 
lambda=lambda(ind);
eVec=eVec(:,ind);
PComp=eVec'*XY;

[U,S,V]=svd(XY'/sqrt(col-1),0); 
lambda2=diag(S).^2;  
sigma =diag(S); 
PComp2=U*XY;

figure(2)
subplot(2,2,1), plot(sigma,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Singular Values')
subplot(2,2,2), plot(lambda2/sum(lambda)*100,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Percentage Variance')
subplot(2,1,2), plot(PComp(1,:))
xlim([0 226])
ylabel('Position')
title('Ideal Case')

figure(4)
plot(PComp2(1,:))
xlim([0 226])
ylabel('Position')
title('Ideal Case')

figure(5)
plot(sigma,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Singular Values')

%% test 2
close all; clear all; clc
load cam1_2.mat;
load cam2_2.mat;
load cam3_2.mat;

[h,w,rgb,frames]=size(vidFrames1_2);
X1=[]; Y1=[];
for j = 1:frames
    img=rgb2gray(vidFrames1_2(:,:,:,j));
    img(:,1:300)=0;
    img(:,410:end)=0; 
    img(1:220,:)=0; 
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X1=[X1 Ix];
    Y1=[Y1 Iy];
end

[h,w,rgb,frames]=size(vidFrames2_2);
X2=[]; Y2=[];
for j=1:frames
    img=rgb2gray(vidFrames2_2(:,:,:,j));
    img(:,1:200)=0;
    img(:,400:end)=0;
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X2=[X2 Ix];
    Y2=[Y2 Iy];
end

[h,w,rgb,frames]=size(vidFrames3_2);
X3=[]; Y3=[];
for j=1:frames
    img=rgb2gray(vidFrames3_2(:,:,:,j));
    img(:,1:250)=0;
    img(:,500:end)=0;
    img(1:210,:)=0;
    img(350:end,:)=0;
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X3=[X3 Ix];
    Y3=[Y3 Iy];
end

Xmin=min([length(X1) length(X2) length(X3)]);
if length(X1) > Xmin
    X1=X1(1:Xmin);
    Y1=Y1(1:Xmin);
end
if length(X2) > Xmin
    X2=X2(1:Xmin);
    Y2=Y2(1:Xmin);
end
if length(X3) > Xmin
    X3=X3(1:Xmin);
    Y3=Y3(1:Xmin);
end

XY=[X1;Y1;X2;Y2;X3;Y3];

figure(1)
for j=1:6
    subplot(3,2,(j)), plot(XY(j,:))
    xlim([0 314])
    ylim([0 500])
    if j==1 || j==2
        title('Camera 1')
    elseif j==3 || j==4
        title('Camera 2')
    elseif j==5 || j==6
        title('Camera 3')
    end
    
    if mod(j,2)==1 %odd
        ylabel('X')
    elseif mod(j,2)==0 %even
        ylabel('Y')
    end
end

[row,col]=size(XY); 
mn=mean(XY,2);
XY=XY-repmat(mn,1,col); 

CovMat=(1/(col-1))*XY*XY';
[eVec,eVal]=eig(CovMat); 
lambda=diag(eVal); 
[B,ind]=sort(-1*lambda);
lambda=lambda(ind);
eVec=eVec(:,ind);
PComp=eVec'*XY; 

figure(2)
plot(PComp(1,:))
xlim([0 314])
ylabel('Position')
title('Noisy Case')

[U,S,V]=svd(XY'/sqrt(col-1),0);
lambda2=diag(S).^2;  
sigma =diag(S); 
PComp2=U*XY; 

figure(2)
subplot(2,2,1), plot(sigma,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Singular Values')
subplot(2,2,2), plot(lambda2/sum(lambda)*100,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Percentage Variance')
subplot(2,1,2), plot(PComp(1,:))
xlim([0 314])
ylabel('Position')
title('Noisy Case')


figure(3)
plot(U(:,1))
xlim([0 314])

figure(4)
plot(PComp2(1,:))
xlim([0 314])
ylabel('Position')
title('Noisy Case')

figure(5)
plot(sigma,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Singular values')
%% test 3
close all; clear all; clc
load cam1_3.mat;
load cam2_3.mat;
load cam3_3.mat;

[h,w,rgb,frames]=size(vidFrames1_3);
X1=[]; Y1=[];
for j=1:frames
    img=rgb2gray(vidFrames1_3(:,:,:,j));
    img(:,1:270)=0; 
    img(:,400:end)=0; 
    img(1:220,:)=0; 
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X1=[X1 Ix]; Y1=[Y1 Iy];
end

[h,w,rgb,frames]=size(vidFrames2_3);
X2=[]; Y2=[];
for j=1:frames
    img=rgb2gray(vidFrames2_3(:,:,:,j));
    img(1:180,:)=0; 
    img(:,1:220)=0;
    img(:,410:end)=0; 
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X2=[X2 Ix]; Y2=[Y2 Iy];
end

[h,w,rgb,frames]=size(vidFrames3_3);
X3=[]; Y3=[];
for j=1:frames
    img=rgb2gray(vidFrames3_3(:,:,:,j));
    img(:,1:250)=0;
    img(:,500:end)=0;
    img(1:190,:)=0;
    img(350:end,:)=0;
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X3=[X3 Ix]; Y3=[Y3 Iy];
end

Xmin=min([length(X1) length(X2) length(X3)]);
if length(X1) > Xmin
    X1=X1(1:Xmin);
    Y1=Y1(1:Xmin);
end
if length(X2) > Xmin
    X2=X2(1:Xmin);
    Y2=Y2(1:Xmin);
end
if length(X3) > Xmin
    X3=X3(1:Xmin);
    Y3=Y3(1:Xmin);
end

XY=[X1;Y1;X2;Y2;X3;Y3];

figure(1)
for j=1:6
    subplot(3,2,(j)), plot(XY(j,:))
    xlim([0 237])
    ylim([0 500])
    if j==1 || j==2
        title('Camera 1')
    elseif j==3 || j==4
        title('Camera 2')
    elseif j==5 || j==6
        title('Camera 3')
    end
    
    if mod(j,2)==1 %odd
        ylabel('X')
    elseif mod(j,2)==0 %even
        ylabel('Y')
    end
end

[row,col]=size(XY); 
mn=mean(XY,2); 
XY=XY-repmat(mn,1,col); 

CovMat=(1/(col-1))*XY*XY'; 
[eVec,eVal]=eig(CovMat); 
lambda=diag(eVal); 
[B,ind]=sort(-1*lambda); 
lambda=lambda(ind);
eVec=eVec(:,ind);
PComp=eVec'*XY; 

figure(2)
plot(PComp(1,:))
xlim([0 237])
ylabel('Position')
title('Horizontal Displacement')

[U,S,V]=svd(XY'/sqrt(col-1),0);
lambda2=diag(S).^2;  
sigma =diag(S); 
PComp2=U*XY; 

figure(2)
subplot(2,2,1), plot(sigma,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Singular Values')
subplot(2,2,2), plot(lambda2/sum(lambda)*100,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Percentage Variance')
subplot(2,1,2), plot(PComp(1,:))
xlim([0 237])
ylabel('Position')
title('Horizontal Displacement')

figure(3)
plot(U(:,1))
xlim([0 237])

figure(4)
plot(PComp2(1,:))
xlim([0 237])
ylabel('Position')
title('Horizontal Displacement')

figure(5)
plot(sigma,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Singular Values')
%% test 4
close all; clear all; clc
load cam1_4.mat;
load cam2_4.mat;
load cam3_4.mat;

[h,w,rgb,frames]=size(vidFrames1_4);
X1=[]; Y1=[];
for j=1:frames
    img=rgb2gray(vidFrames1_4(:,:,:,j));
    img(:,1:320)=0;
    img(:,460:end)=0;
    img(1:200,:)=0;
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X1=[X1 Ix]; Y1=[Y1 Iy];
end

[h,w,rgb,frames]=size(vidFrames2_4);
X2=[]; Y2=[];
for j=1:frames
    img=rgb2gray(vidFrames2_4(:,:,:,j));
    img(:,1:220)=0;
    img(:,420:end)=0;
    img(1:110,:)=0;
    img(390:end,:)=0;
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X2=[X2 Ix]; Y2=[Y2 Iy];
end

[h,w,rgb,frames]=size(vidFrames3_4);
X3=[]; Y3=[];
for j=1:frames
    img=rgb2gray(vidFrames3_4(:,:,:,j));
    img(:,1:290)=0;
    img(:,550:end)=0;
    img(1:150,:)=0;
    img(320:end,:)=0;
    [M,I]=max(img(:));
    [Iy,Ix]=ind2sub(size(img),I);
    X3=[X3 Ix]; Y3=[Y3 Iy];
end

Xmin=min([length(X1) length(X2) length(X3)]);
if length(X1) > Xmin
    X1=X1(1:Xmin);
    Y1=Y1(1:Xmin);
end
if length(X2) > Xmin
    X2=X2(1:Xmin);
    Y2=Y2(1:Xmin);
end
if length(X3) > Xmin
    X3=X3(1:Xmin);
    Y3=Y3(1:Xmin);
end

XY=[X1;Y1;X2;Y2;X3;Y3];

figure(1)
for j=1:6
    subplot(3,2,(j)), plot(XY(j,:))
    xlim([0 392])
    ylim([0 500])
    if j==1 || j==2
        title('Camera 1')
    elseif j==3 || j==4
        title('Camera 2')
    elseif j==5 || j==6
        title('Camera 3')
    end
    
    if mod(j,2)==1 %odd
        ylabel('X')
    elseif mod(j,2)==0 %even
        ylabel('Y')
    end
end

XY = [X1;Y1;X2;Y2;X3;Y3];
[row,col]=size(XY);
mn=mean(XY,2); 
XY=XY-repmat(mn,1,col); 

CovMat=(1/(col-1))*XY*XY'; 
[eVec,eVal]=eig(CovMat); 
lambda=diag(eVal); 
[B,ind]=sort(-1*lambda); 
lambda=lambda(ind);
eVec=eVec(:,ind);
PComp=eVec'*XY; 


figure(2)
plot(PComp(1,:))
ylabel('Position')
title('Horizontal Displacement and Rotation')

[U,S,V]=svd(XY'/sqrt(col-1),0); 
lambda2=diag(S).^2;  
sigma =diag(S); 
PComp2=U*XY; 

figure(2)
subplot(2,2,1), plot(sigma,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Singular Values')
subplot(2,2,2), plot(lambda2/sum(lambda)*100,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Percentage Variance')
subplot(2,1,2), plot(PComp(1,:))
xlim([0 392])
ylabel('Position')
title('Horizontal Displacement and Rotation')


figure(3)
plot(U(:,1))
xlim([0 392])

figure(4)
plot(PComp2(1,:))
xlim([0 392])
ylabel('Position')
title('Horizontal Displacement and Rotation')

figure(5)
plot(sigma,'-o','Linewidth',[1])
axis([1 6 0 100])
title('Singular Values')