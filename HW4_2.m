close all; clear; clc

[TrainImgs, TrainLabels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[TestImgs, TestLabels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
[training_rows,training_cols,training_images]=size(TrainImgs);
[testing_rows,testing_cols,testing_images]=size(TestImgs);
for i=1:training_images
    training_data(:,i)=double(reshape(TrainImgs(:,:,i),training_rows*training_cols,1));
end
for i=1:testing_images
    testing_data(:,i)=double(reshape(TestImgs(:,:,i),testing_rows*testing_cols,1));
end
%% SVD
[U,S,V]=svd(training_data,'econ');
sigma =diag(S); lambda2=diag(S).^2; 
digits=(S*V')';

figure(1) %display the first 10 columns of U (PCA modes)
for j=1:10
    subplot(2,5,j)
    ut1=reshape(U(:,j),28,28);
    ut2=ut1(28:-1:1,:);
    pcolor(ut2)
    set(gca,'Xtick',[],'Ytick',[])
end

figure(2) %singular values
subplot(2,1,1), plot(sigma/sum(sigma),'ko-','Linewidth',[1])
axis([0 800 0 0.07])
title('Singular Values')
subplot(2,1,2), plot(lambda2/sum(lambda2)*100,'ko-','Linewidth',[1])
axis([0 800 0 50])
title('Percentage Variance')
%% V-modes
figure(3);
for i=0:9
    ind = find(TrainLabels==i);
    scatter3(V(ind,2),V(ind,3),V(ind,5),20,TrainLabels(ind),'.')
    hold on
end
xlabel('V(:,2)'),ylabel('V(:,3)'),zlabel('V(:,5)')
legend({'0','1','2','3','4','5','6','7','8','9'});
colormap hsv

% LDA
feature=50;
train0=digits(TrainLabels==0,1:feature);
train1=digits(TrainLabels==1,1:feature);
%train5=digits(TrainLabels==7,1:feature);
[row0,col0]=size(train0);
[row1,col1]=size(train1);
%[row5,col5]=size(train5);
train_vec=[train0; train1];
group1=[0*ones(row0,1); 1*ones(row1,1)];
PComp=(U'*testing_data)';
test0=PComp(TestLabels==0,1:feature);
test1=PComp(TestLabels==1,1:feature);
%test5=PComp(TestLabels==7,1:feature);
[row0,col0]=size(test0);
[row1,col1]=size(test1);
%[row5,col5]=size(test5);
test1=[test0; test1];
group2=[0*ones(row0,1); 1*ones(row1,1)];
prediction=classify(test1,train_vec,group1);
error=sum(abs(group2-prediction)>0);
accuracy=1-error/length(group2);

figure(5)
bar(prediction)
ylabel('Prediction (0 or 1)')
% LDA
feature=50;
train0=digits(TrainLabels==0,1:feature);
train2=digits(TrainLabels==2,1:feature);
train8=digits(TrainLabels==8,1:feature);
[row0,col0]=size(train0);
[row2,col2]=size(train2);
[row8,col8]=size(train8);
train_vec=[train0; train2; train8];
group1=[0*ones(row0,1); 2*ones(row2,1); 8*ones(row8,1)];
PComp=(U'*testing_data)';
test0=PComp(TestLabels==0,1:feature);
test2=PComp(TestLabels==2,1:feature);
test8=PComp(TestLabels==8,1:feature);
[row0,col0]=size(test0);
[row2,col2]=size(test2);
[row8,col8]=size(test8);
test1=[test0; test2; test8];
group2=[0*ones(row0,1); 2*ones(row2,1); 8*ones(row8,1)];
prediction=classify(test1,train_vec,group1);
error=sum(abs(group2-prediction)>0);
accuracy=1-error/length(group2);
figure(5)
bar(prediction)
ylabel('Prediction (0, 2, or 8)')
% Classifier Tree
train1=digits(:,1:feature);
PComp=(U'*testing_data)';
test1=PComp(:,1:feature);
tree = fitctree(train1,TrainLabels,'OptimizeHyperparameters','auto');
predection=predict(tree,test1);
error=sum(abs(TestLabels-predection)>0);
accuracy=1-error/length(TestLabels);
% SVM
train1=digits(:,1:feature)/max(max(S));
test1=PComp(:,1:feature)/max(max(S));
for j = 0:9
    ind = TrainLabels==j;
    models{j+1} = fitcsvm(train1,ind,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end
for j = 0:9
    [~,pred] = predict(models{j},test1);
    acu(:,j) = pred(:,2); 
end
[~,M] = max(acu,[],2);
error=sum(abs(TestLabels+1-M)>0);
accuracy=1-error/length(TestLabels)