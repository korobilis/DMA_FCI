function [y,mm,vv] = standardize2(x)

% Function to make your data have mean 0 and variance 1.
% Data in x are Txp, i.e. T time series observations times p variables

mm = mean(x,1);
vv = std(x);
t = size(x,1);
y = (x - repmat(mm,t,1))./repmat(vv,t,1);