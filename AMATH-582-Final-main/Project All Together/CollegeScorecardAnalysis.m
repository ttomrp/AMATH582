clear all; close all; clc;

%%%%Final Project - Exploratory Data Analysis%%%%
%College Scorecard Data 2016-17

scorecardTable = readtable('MERGED2016_17_CLEAN.csv');
data = table2array(scorecardTable(:,3:end))';
[m,n] = size(data);

%% Perform SVD and Analyze the Spectrum:
[u,s,v] = svd(data,'econ');

figure(1)
plot(diag(s)/sum(diag(s)),'ko','Linewidth',2)
title('Normalized Singular Value Spectrum')
xlabel('Principal Component'), ylabel('Energy')
set(gca,'FontSize',14)

rank = 7;
ured = u(:,1:rank); 
sred = s(1:rank,1:rank); 
vred = v(:,1:rank);

weight = diag(sred)/sum(diag(sred));
varweight = (weight.*ured')';

figure(2)
bar(1:m,abs(mean(varweight,2))/max(abs(mean(varweight,2))))
xticks(0:25)
xticklabels(string(2:27))
title("Energy Weighted Variable Importance for PC's")
xlabel('Variable from Dataset')
ylabel('Relative Weight in PCAs')
set(gca,'FontSize',14)

%Variables in this data set by order of importance:
%1: #17 - COSTT4_A, average cost of attendance
%2: #25 - DEBT_MDN, median loan principal upon entering repayment
%3: #7  - UGDS, number of undergraduate students enrolled
%4: #18 - AVGFACSAL, average faculty salary
%5: #6 - SAT_AVG, SAT test average score (nearly 0 weight)

%2-D Projection of V-modes (two clear clusters)
figure(3)
plot(vred(:,2),vred(:,3),'o','Linewidth',1)
title('Projection of Principal Components for Schools')
xlabel('Principal Component 2'), ylabel('Principal Component 3'),
set(gca,'FontSize',12)
hold off

%% Attempt to Cluster
%NOTE: This section was done prior to analyzing the principal components
%and is essentially uninformed exploration.

%Index labels by state (different data set used orginally with 15
%western states):
j = 1;
idxState = zeros(52,1);
for i = 1:(n-1)
    if string(scorecardTable.STABBR(i+1)) ~= string(scorecardTable.STABBR(i))
        idxState(j,1) = i+1;
        j = j+1;
    end
end
idxState = [1; idxState];

%Index labels by gradutation rates:
lblGrad = zeros(n,1);
for i = 1:n
    if data(17,i)<0.50
        lblGrad(i) = 1;
    elseif data(17,i)>=0.50
        lblGrad(i) = 2;
    end
end

%Sort data by grad rates:
[~,lblGradIdx] = sort(lblGrad);
lblGrad = lblGrad(lblGradIdx);
dataGradSort = data(:,lblGradIdx);
lblGradLoc = [1, nnz(lblGrad==1), nnz([lblGrad==1,lblGrad==2])];

gradRates = {'0 to 50%','50 to 100%'};

% 2-D v-mode projections again, attempt to cluster by grad rates:
for i = 1:2
    figure(4)
    plot(vred(lblGradLoc(i):lblGradLoc(i+1),2),...
         vred(lblGradLoc(i):lblGradLoc(i+1),3),'o','Linewidth',1)
    hold on
    grid on
end

legend(gradRates,'Location','bestoutside')
title('Projection of Schools Clustered by Graduation Rates')
xlabel('Principal Component 2'), ylabel('Principal Component 3'),
set(gca,'FontSize',12)
hold off

%v23 plots shows us that grad rates clustering is likely not a good choice.

%% Unsupervised Clustering Attempt (Grad Rates)
%Since the chosen labels above appears not to relate to the clusters
%that were visualized, see if GMM can sort these:
iters = 100;
for i = 1:iters
    q1 = randperm(nnz(lblGrad==1)); %randperm creates a random shuffling of ints
    q2 = randperm(nnz(lblGrad==2));

    x50 = vred(1:lblGradLoc(2),:); 
    x100 = vred(lblGradLoc(2):lblGradLoc(3),:);

    %Split into training and test set
    split = 5/8;
    strain50 = floor(nnz(lblGrad==1)*split);
    strain100 = floor(nnz(lblGrad==2)*split);

    %train:
    xtrain = [x50(q1(1:strain50),:); x100(q2(1:strain100),:)];

    %test:
    xtest = [x50(q1(strain50+1:end),:); x100(q2(strain100+1:end),:)];

    %labels:
    ctrain = [ones(strain50,1); 2*ones(strain100,1)];

    gm = fitgmdist(xtrain,2);
    pre(:,i) = cluster(gm,xtest);
end

%Show results:
figure(5)
histogram(pre,'Normalization','probability')
title({'GMM Predictions - Unsupervised (',num2str(iters,'%02d'),' iterations)'})
xticks([1 2 3 4])
xticklabels(gradRates)
ylabel('Probability')
axis([0 3 0 1])
set(gca,'FontSize',12)

%<50% = ~500 and >50% = ~1200, not 50/50 like plot shows (random guessing)
%Unsupervised clustering did not work for grad rates, again showing that
%the cluster pattern is predominantly based on something else.

%% Next Steps: Attendance Cost Exploration
%Interesting plot - two distinct groups of cost, each has a higher grad rate
%at upper end and low grad rate at lower end:
figure(6)
plot(data(15,:),data(17,:)*100,'o'), hold on
plot((1:n)*max(data(15,:))/n,3/10*(1:n)-105,'r','LineWidth',2)
title('Impact of Attendance Cost on Graduation Rates')
ylabel('Percent Graduated')
xlabel('Cost, $')
axis([0 max(data(15,:))*1.05 0 100])
set(gca,'FontSize',14)
text(24000,7,['Manually placed separator'],'FontSize',14)
hold off

%Index labels by attendance cost:
lblCost = zeros(n,1);
for i = 1:n
    if data(15,i)<32000
        lblCost(i) = 1;
    elseif data(15,i)>=32000
        lblCost(i) = 2;
    end
end

%Sort data by attendance cost:
[~,lblCostIdx] = sort(lblCost);
lblCost = lblCost(lblCostIdx);
dataCostSort = data(:,lblCostIdx);
lblCostLoc = [1, nnz(lblCost==1), nnz([lblCost==1,lblCost==2])];

%% Supervised Clustering (Attendance Cost)
iters = 1000;
for i = 1:iters
    q1 = randperm(nnz(lblCost==1)); %randperm creates a random shuffling of ints
    q2 = randperm(nnz(lblCost==2));

    x1 = vred(1:lblCostLoc(2),:); 
    x2 = vred(lblCostLoc(2):lblCostLoc(3),:);

    %Split into training and test set
    split = 5/8;
    strain1 = floor(nnz(lblCost==1)*split);
    strain2 = floor(nnz(lblCost==2)*split);
    stest1 = floor(nnz(lblCost==1)*(1-split));
    stest2 = floor(nnz(lblCost==2)*(1-split));

    %train:
    xtrain = [x1(q1(1:strain1),:); x2(q2(1:strain2),:)];

    %test:
    xtest = [x1(q1(strain1+1:end),:); x2(q2(strain2+1:end),:)];

    %labels:
    ctrain = [ones(strain1,1); 2*ones(strain2,1)];
    ctest = [ones(stest1+1,1); 2*ones(stest2+1,1)];

    %Supervised 1 - Naive Bayes
    nb = fitcnb(xtrain,ctrain);
    preNB(:,i) = predict(nb,xtest);
    
    accNB(i) = (1-sum(abs(preNB(:,i)-ctest))/length(xtest))*100;

    %Supervised 2 - Linear Discriminant Analysis (LDA)
    LDA = fitcdiscr(xtrain,ctrain);
    preLDA(:,i) = predict(LDA,xtest);
    
    accLDA(i) = (1-sum(abs(preLDA(:,i)-ctest))/length(xtest))*100;
end

%Accuracy:
attndCost = {'< $32000','> $32000'};

figure(6)
subplot(1,2,1)
bar(mean(preNB,2))
title({'Naive Bayes - 1000 iters'})
xticks(size(preNB,1)*[1/4 3/4])
xticklabels(attndCost)
ylabel('Probability')
set(gca,'FontSize',14)
text(length(xtest)/35,1.8,['Accuracy =  ',...
     num2str(mean(accNB),'%.2f'),'%'],'FontSize',14)

subplot(1,2,2)
bar(mean(preLDA,2))
title({'LDA - 1000 iters'})
xticks(size(preNB,1)*[1/4 3/4])
xticklabels(attndCost)
ylabel('Probability')
set(gca,'FontSize',14)
text(length(xtest)/35,1.8,['Accuracy =  ',...
     num2str(mean(accLDA),'%.2f'),'%'],'FontSize',14)
 
%Good accuracy with both methods at separating, given cost comparisons as
%inputs. Further study could import other years of data to use as a test
%case.
 







