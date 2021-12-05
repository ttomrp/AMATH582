function NBFunc(data, titleName)

    %data has two majors in it
    meanDebt = mean(data.P_DEBTMEAN);
    stdDebt = std(data.P_DEBTMEAN);
    stand_P_DEBTMEAN = (data.P_DEBTMEAN - meanDebt) / stdDebt;
    data.P_DEBTMEAN = stand_P_DEBTMEAN; 
    
    meanEarn = mean(data.P_MD_EARN_WNE);
    stdEarn = std(data.P_MD_EARN_WNE);
    stand_P_MD_EARN_WNE = (data.P_MD_EARN_WNE - meanEarn) / stdEarn;
    data.P_MD_EARN_WNE = stand_P_MD_EARN_WNE; 
    
    debt = min(data.P_DEBTMEAN) : 0.01 : max(data.P_DEBTMEAN);
    earn = min(data.P_MD_EARN_WNE) : 0.01 : max(data.P_MD_EARN_WNE);
    
    % Create classification model from data
    model = fitcnb(data, 'P_MAJORID~P_DEBTMEAN+P_MD_EARN_WNE');
    
    [d, e] = meshgrid(debt, earn);
    ms = predict(model, [d(:) e(:)]);
    
    color = lines(6); % Generate color values
    gscatter(d(:), e(:), ms, color(4:6,:));
    
    hold on;  
    
    title(titleName);  
    xlabel('Debts');
    ylabel('Earnings');
    
    gscatter(data.P_DEBTMEAN, data.P_MD_EARN_WNE, data.P_MAJOR, color(1:6,:), '.', 8);
end