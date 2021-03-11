close all; clear all; clc
[TrainImages, TrainLabels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
trainingData = double(reshape(TrainImages, size(TrainImages,1)*size(TrainImages,2), []));
trainingLab = double(TrainLabels);

%SVD over the whole training dataset 
[U,S,V]=svd(trainingData,'econ');
lambda2=diag(S).^2;  
sigma =diag(S); 
PComp=U*trainingData;
digits=S*V';
%display the first 10 columns of U (PCA modes)
figure(1)
for j=1:10
    subplot(2,5,j)
    ut1=reshape(U(:,j),28,28);
    ut2=ut1(28:-1:1,:);
    pcolor(ut2)
    set(gca,'Xtick',[],'Ytick',[])
end
figure(2)
subplot(2,1,1), plot(sigma,'ko','Linewidth',[1])
axis([1 50 0 500000])
%xlim([1 10])
title('Singular Values')
subplot(2,1,2), plot(lambda2/sum(lambda2)*100,'ko','Linewidth',[1])
axis([1 50 0 100])
%xlim([0 10])
title('Percentage Variance')

Ind0=find(trainingLab==0);
Ind1=find(trainingLab==1);
Ind2=find(trainingLab==2);

for j=1:5000 %take the first 5000 images
    ind0=Ind0(j);
    Matrix0(:,j)=trainingData(:,ind0);
    ind1=Ind1(j);
    Matrix1(:,j)=trainingData(:,ind1);
    ind2=Ind2(j);
    Matrix2(:,j)=trainingData(:,ind2);
end

feature=50;
trainMat=[Matrix0,Matrix1,Matrix2]; 
%trainMat=[Matrix0,Matrix1]; 
[U,S,V]=svd(trainMat,'econ');
lambda2=diag(S).^2;  
sigma=diag(S); 
digits=S*V';
U=U(:,1:feature);
Digit0=digits(1:feature,1:5000);
Digit1=digits(1:feature,5001:10000);
Digit2=digits(1:feature,10001:end);

%display the first 10 columns of U (PCA modes)
for j=1:4
    subplot(2,2,j)
    ut1=reshape(U(:,j),28,28);
    ut2=ut1(28:-1:1,:);
    pcolor(ut2)
    set(gca,'Xtick',[],'Ytick',[])
end

figure(2)
subplot(2,1,1), plot(sigma,'ko','Linewidth',[1])
axis([1 50 0 200000])
title('Singular Values')
subplot(2,1,2), plot(lambda2/sum(lambda2)*100,'ko','Linewidth',[1])
axis([1 50 0 100])
title('Percentage Variance')

figure(3)
for j=1:3
    subplot(3,2,2*j-1)    
    plot(1:100,V(1:100,j),'k-')
    title('Zeros')
    set(gca,'XTick',[])
    subplot(3,2,2*j)
    plot(5001:5100,V(5001:5100,j),'k-')
    title('Ones')
    set(gca,'XTick',[])
    %subplot(3,3,3*j)
    %plot(10001:10100,V(10001:10100,j),'k-')
    %title('Twos')
end

%LDA
m0=mean(Digit0,2);
m1=mean(Digit1,2);
%m2=mean(Digit2,2);

Sw=0; %within class variance
for i=1:5000
    Sw=Sw+(Digit0(:,i)-m0)*(Digit0(:,i)-m0)';
end
for i=1:5000
    Sw=Sw+(Digit1(:,i)-m1)*(Digit1(:,i)-m1)';
end
%for i=1:5000
%    Sw=Sw+(Digit2(:,i)-m2)*(Digit2(:,i)-m2)';
%end
%Sb=(m0-m1-m2)*(m0-m1-m2)'; %between clas varaince
Sb=(m0-m1)*(m0-m1)';

[V2,D] = eig(Sb,Sw); % linear discriminant analysis
[lambda,ind] = max(abs(diag(D)));
w = V2(:,ind); w = w/norm(w,2);
v0 = w'*Digit0; v1 = w'*Digit1; %v2=w'*Digit2
result = [v0,v1];
if mean(v0)>mean(v1)
    w=-w; v0=-v0; v1=-v1;
end
sort0=sort(v0);
sort1=sort(v1);

t1 = length(sort0);
t2 = 1;
while sort0(t1)>sort1(t2)
    t1 = t1-1;
    t2 = t2+1;
end
threshold = (sort0(t1)+sort1(t2))/2;

figure(4)
subplot(2,1,1)
hist(sort0,30); hold on, plot([threshold threshold],[0 800],'r')
%set(gca,'Xlim',[-200 200],'Ylim',[0 10],'Fontsize',[14])
title('Zero')
subplot(2,1,2)
hist(sort1,30,'r'); hold on, plot([threshold threshold],[0 800],'r')
%set(gca,'Xlim',[-200 200],'Ylim',[0 10],'Fontsize',[14])
title('One')