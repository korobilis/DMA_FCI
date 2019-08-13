function [a_prior,V_prior] = Minn_prior_KOOP(gamma,M,p,K)

% This is the version of the Minnesota prior with no dependence on the
% standard deviations of the univariate regressions. This prior allows
% online estimation and forecasting of the large TVP-VAR.

% 1. Minnesota Mean on VAR regression coefficients
A_prior = [0.9*eye(M); zeros((p-1)*M,M)]';
a_prior = (A_prior(:));

% 2. Minnesota Variance on VAR regression coefficients

% Create an array of dimensions K x M, which will contain the K diagonal   
% elements of the covariance matrix, in each of the M equations.
V_i = zeros(K/M,M);

for i = 1:M  % for each i-th equation
    for j = 1:K/M   % for each j-th RHS variable        
        V_i(j,i) = gamma./((ceil(j/M)).^2); % variance on own lags           
        % Note: the "ceil((j-1)/M^2)" command finds the associated lag 
        % number for each parameter         
    end
end

% Now V (MINNESOTA VARIANCE) is a diagonal matrix with diagonal elements the V_i'  
V_i_T = V_i';
V_prior = diag(V_i_T(:));  % this is the prior variance of the vector alpha
