clear all;clc

data = readtable('FieldOfStudyData1516_1617_PP.csv');

% P_MD_EARN_WNE = Median earnings of federally-aided completers in the earnings cohort
% MD_EARN_WNE = Median earnings of federally-aided completers in the earnings cohort	programs
% OPEID6 = 6-digit OPE ID for institution	programs	
% CIPCODE = Classification of Instructional Programs (CIP) code for the field of study	programs	
% P_CREDLEV = Level of credential

xdata = data;
toDelete = xdata.P_DEBTMEAN == "PrivacySuppressed";
xdata(toDelete,:) = [];
xdata.P_DEBTMEAN = str2double(xdata.P_DEBTMEAN);

toDelete = xdata.P_MD_EARN_WNE == "PrivacySuppressed";
xdata(toDelete,:) = [];
xdata.P_MD_EARN_WNE = str2double(xdata.P_MD_EARN_WNE);

writetable(xdata,'DataWithDebtAndEarningValues.csv');
