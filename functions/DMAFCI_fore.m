function [y_f,PL_1] = DMAFCI_fore(ydata,beta_t,factors,Q_t,p,nfore)

y_f = zeros(nfore,p);
PL_1 = zeros(p,1);
for ii = 1:nfore               
    FORECASTS = (beta_t^ii)*factors;               
    %y_fore(ii,:,nmod) = Ymeans + Ystds.*FORECASTS(1:p,:)';
    y_f(ii,:) = FORECASTS(1:p,:)';                         
end
% 2/ 1-step ahead predictive likelihood of each variable              
for ii_5 = 1:p       
    PL_1(ii_5,1) = normpdf(ydata(:,ii_5),y_f(1,ii_5),Q_t(ii_5,ii_5));
end
