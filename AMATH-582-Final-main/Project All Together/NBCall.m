clc;clear all;close all;
%warning off;

% P_MAJORID P_MAJOR (general, poorly-defined categories for certain major types)
% 1 Agriculture
% 2 Arts (check)
% 3 Business (check)
% 4 Communication
% 5 Computer Science (check)
% 6 Construction
% 7 Education (check)
% 8 Engineering (check)
% 9 Information
% 10 Language
% 11 Law
% 12 Math 
% 13 Medical 
% 14 Nature
% 15 Social Science/Studies (check)
% 16 Sciences 
MajorList=[2,3,5,7,8,12,13,15,16,17];

data = readtable('DataWithDebtAndEarningValuesTEST.csv');

%removes degrees that are not bachelors(3) or masters(5) or doctors(6)
%toDelete = (data.P_CREDLEV ~= 3 & data.P_CREDLEV ~= 5 & data.P_CREDLEV ~= 6); 
toDelete = (data.P_CREDLEV ~= 3 & data.P_CREDLEV ~= 5); 
data(toDelete,:) = [];


% Make new matrix of just two major categories
MajorData = data; %currently has everything in it
major1=3; 
major2=15; 
% remove every row that isn't one of the two defined major categories
toDelete = (MajorData.P_MAJORID ~= major1 ...
    & MajorData.P_MAJORID ~= major2); 
MajorData(toDelete,:) = [];
% now MajorData just has data for two major categories

%pass in matrix of two majors
NBFunc(MajorData, 'Business vs Social Science');


    