% P_CIPCODE	P_CIPDESC
% 2701	Mathematics.
% 2703	Applied Mathematics.
% 5202	Business Administration, Management and Operations.
% 1107	Computer Science.
% 2301	English Language and Literature, General.
% 1408	Civil Engineering.
% 2201	Law.
% 2705	Statistics.

clc;
clear all;
close all;
warning off;

data = readtable('DataWithDebtAndEarningValues.csv');

toDelete = (data.P_CREDLEV ~= 3 & data.P_CREDLEV ~= 5);
data(toDelete,:) = [];

% % Applied Mathematics
% AppliedMathematics = data;
% toDelete = AppliedMathematics.P_CIPCODE ~= 2703;
% AppliedMathematics(toDelete,:) = [];
% 
% NaiveBayesFigureFunction(AppliedMathematics, 'Applied Mathematics Classification');

% Computer Science
ComputerScience = data;
toDelete = ComputerScience.P_CIPCODE ~= 1107;
ComputerScience(toDelete,:) = [];

NaiveBayesFigureFunction(ComputerScience, 'Computer Science Classification');
% 
% EnglishLanguageAndLiterature = data;
% toDelete = EnglishLanguageAndLiterature.P_CIPCODE ~= 2301;
% EnglishLanguageAndLiterature(toDelete,:) = [];
% 
% NaiveBayesFigureFunction(EnglishLanguageAndLiterature, 'English Language and Literature, General');

