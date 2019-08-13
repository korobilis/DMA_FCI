function y = standardize_miss(x)

% Function to make your data have mean 0 and variance 1.
% Data in x are Txp, i.e. T time series observations times p variables
% This version treats missing values in the sample (usually begining or
% end), and standardizes the nonmissing values. Missing values are retained 
% to NaN.
y=0*x;

% find how many variables (columns) are in x
n = size(x,2);

% Start
for i = 1:n
    f = find(1-isnan(x(:,i)));
    y(f,i) = (x(f,i) - mean(x(f,i)))./(std(x(f,i)));
end
