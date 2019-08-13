% Competing FCIs - Forecasts from competing models
% To minimize programming error, I treat the competing FCIs as FAVAR models with a known factor (FCI),
% hence I am using the same code as the FAVAR models.
%-----------------------------------------------------------------------------------------
% The model is:
%     _    _     _              _     _    _     _    _
%    | y[t] |   |   I        0   |   | y[t] |   |   0  |
%    |      | = |                | x |      | + |      |
%	 | x[t] |   | L[y,t]  L[f,t] |   | f[t] |   | e[t] |
%     -    -     -              -     -    -     -    -
%	 
%     _    _              _      _
%    | y[t] |            | y[t-1] |   
%    |      | = B[t-1] x |        | + u[t]
%    | f[t] |            | f[t-1] |   
%     -    -              -      -     
% where L[t] = (L[y,t] ; L[f,t]) and B[t] are coefficients, f[t] are factors, e[t]~N(0,V[t])
% and u[t]~N(0,Q[t]), and
% 
%   L[t] = L[t-1] + v[t]
%   B[t] = B[t-1] + n[t]
%
% with v[t]~N(0,H[t]), n[t]~N(0,W[t])
%
% All covariances follow EWMA models of the form:
%
%  V[t] = l_1 V[t-1] + (1 - l_1) e[t-1]e[t-1]'
%  Q[t] = l_2 Q[t-1] + (1 - l_2) u[t-1]u[t-1]'
%
% with l_1, l_2, l_3 and l_4 being the decay/forgetting factors (see paper for details).
%-----------------------------------------------------------------------------------------
% Written by Dimitris Korobilis
% University of Glasgow
% This version: 08 July, 2013
%-----------------------------------------------------------------------------------------

clear all;
close all;
clc;

% Add path of data and functions
addpath('data');
addpath('functions');

%-------------------------------USER INPUT--------------------------------------
% Model specification
nfac = 1;         % number of factors
nlag = 4;         % number of lags of factors

% Control the amount of variation in the measurement and error variances
l_1 = 0.96;       % Decay factor for measurement error variance
l_2 = 0.96;       % Decay factor for factor error variance
l_3 = 0.99;          % Forgetting factor for loadings error variance
l_4 = 0.99;       % Forgetting factor for VAR coefficients error variance

% Select if y[t] should be included in the measurement equation (if it is
% NOT included, then the coefficient/loading L[y,t] is zero for all periods
y_true = 1;       % 1: Include y[t]; 0: Do not include y[t]

% Select transformations of the macro variables in Y
transf = 1;       % 1: Use first (log) differences (only for CPI, GDP, M1)
                  % 2: Use annualized (CPI & GDP) & second (log) differences (M1)                   
                  
% Select a subset of the 6 variables in Y                  
subset = 6;       % 1: Infl.- GDP - Int. Rate (3 vars)
                  % 2: Infl. - Unempl. - Int. Rate (3 vars)
                  % 3: Infl. - Inf. Exp. - GDP - Int. Rate (4 vars)
                  % 4: Infl. - Inf. Exp. - GDP - M1 - Int. Rate (5 vars)
                  % 5: Infl. - Inf. Exp. - Unempl. - M1 - Int. Rate (5 vars)
                  % 6: Infl. - GDP - Unempl. - M1 - Int. Rate (5 vars)

% Forecasting
nfore = 4;        % Forecast horizon (note: forecasts are iterated)
t0 = '2000Q1';    % Set initial estimation period
              
%----------------------------------LOAD DATA----------------------------------------
% Load Koop and Korobilis (2012) quarterly data
% load data used to extract factors
load xdata.dat;
% load data on inflation, gdp and the interest rate 
load ydata.dat;    % first log differences (inflation/gpd)
load ydata2.dat;   % second log differences (inflation/gpd)
% load data on FCIs estimated by others (bloomberg, federal reserve banks etc)
load other_FCIs.dat;
% load transformation codes (see file transx.m)
load tcode.dat;
% load the file with the dates of the data (quarters)
load yearlab.mat;
% load the file with the names of the variables
load varnames.mat;
namesXY = ['Inflation' ; 'Infl. Exp.'; 'GDP'; 'Unemployemnt'; 'M1'; 'FedFunds'; varnames ];

if transf == 2
    ydata = ydata2;
end

% Select subsect of vector Y
if subset == 1
    ydata = ydata(:,[1 3 6]); namesXY = namesXY([1 3 6:size(namesXY,1)]);
elseif subset == 2
    ydata = ydata(:,[1 4 6]); namesXY = namesXY([1 4 6:size(namesXY,1)]);
elseif subset == 3
    ydata = ydata(:,[1 2 3 6]); namesXY = namesXY([1 2 3 6:size(namesXY,1)]);
elseif subset == 4
    ydata = ydata(:,[1 2 3 5 6]); namesXY = namesXY([1 2 3 5 6:size(namesXY,1)]);
elseif subset == 5
    ydata = ydata(:,[1 2 4 5 6]); namesXY = namesXY([1 2 4 5 6:size(namesXY,1)]);
elseif subset == 6
    ydata = ydata(:,[1 3 4 5 6]); namesXY = namesXY([1 3 4 5 6:size(namesXY,1)]);    
end

% Convert t0 to numeric value
t0=find(strcmp(yearlab,t0)==1);

% Demean and standardize data (needed to extract Principal Components)
xdata = standardize_miss(xdata) + 1e-40 ;
xdata(isnan(xdata)) = 0;
% ydata = standardize(ydata);

% Define X and Y matrices
X = xdata;   % X contains the 'xdata' which are used to extract factors.
Y = ydata;   % Y contains inflation, gdp and the interest rate

% Number of observations and dimension of X and Y
t = size(Y,1); % t time series observations
n = size(X,2); % n series from which we extract factors
p = size(Y,2); % and p macro series
r = nfac + p;  % number of factors and macro series
q = n + p;     % number of observed and macro series

% Set dimensions of useful quantities
m = nlag*(r^2);  % number of VAR parameters
k = nlag*r;         % number of sampled factors

% ======================| Form all possible model combinations |======================
KK = size(other_FCIs,2);  % Number of models is number of FCIs

% =========================| PRIORS |================================
% Initial condition on the factors
factor_0.mean = zeros(k,1);
factor_0.var = 10*eye(k);
% Initial condition on lambda_t
lambda_0.mean = zeros(q,r);
lambda_0.var = 1*eye(r);
% Initial condition on beta_t
[b_prior,Vb_prior] = Minn_prior_KOOP(0.1,r,nlag,m); % Obtain a Minnesota-type prior
beta_0.mean = b_prior;
beta_0.var = Vb_prior;
% Initial condition on the covariance matrices
V_0 = 0.1*eye(q); V_0(1:p,1:p) = 0;
Q_0 = 0.1*eye(r);

% Put all decay/forgetting factors together in a vector
l = [l_1; l_2; l_3; l_4];

% Initialize matrix of forecasts
f_l = zeros(t,KK);
y_fore = zeros(nfore,p,KK);
y_t_other = zeros(nfore,p,KK,t-t0);
Yraw_f = [ydata ; NaN(nfore,p)];
PL = zeros(p,KK);

MAFE_other = zeros(t-t0,p,nfore,KK);
MSFE_other = zeros(t-t0,p,nfore,KK);
PL_other = zeros(t-t0,p,KK);

%======================= BEGIN KALMAN FILTER ESTIMATION =======================
tic;

for irep = t0+1:t-1
    if mod(irep,ceil(t./40)) == 0
        disp([num2str(100*(irep/t)) '% completed'])       
        toc;   
    end
    
    % Standardize data up to time irep (i.e. using all available
    % information at time irep)
    X_st = standardize_miss(xdata(1:irep,:));% + 1e-20;   
    X_st(isnan(X_st)) = 0;   
    %[Y,Ymeans,Ystds] = standardize2(ydata(1:irep,:));      
    Y = ydata(1:irep,:);    
     
    
    % Iterate over all models
    for nmod = 1:KK
        % Extract Principal Components using data up to time irep
        X = X_st;
        F = standardize_miss(other_FCIs(1:irep,nmod));
        F(isnan(F))=0;
        YX = [Y,X];
        YF = [Y,F];
        
        % Estimate the FCI using the method in Koop and Korobilis (2013):
        % ====| STEP 1: Update Parameters Conditional on PC 
        [beta_t,beta_new,lambda_t,V_t,Q_t] = KFS_parameters(YX,YF,l,nfac,nlag,y_true,k,m,p,q,r,irep,lambda_0,beta_0,V_0,Q_0);
        
        % Create lags on the (FA) VAR part
        YFn = YF;
        YY = YFn(nlag+1:end,:); XX = mlag2(YFn,nlag); XX = XX(nlag+1:end,:);       
        %=========
        if irep>t0 && irep<t
            % 1/ Forecast of model nmod, period irep
            factors = [YFn(end,:) XX(end,1:end-r)]';
            [y_f,PL_1] = DMAFCI_fore(ydata(irep+1,:),beta_t(:,:,irep),factors,Q_t(:,:,irep),p,nfore);
            y_fore(:,:,nmod) = y_f;
            PL(:,nmod) = PL_1;
        end        
    end
    
    if irep>t0
        for nmod = 1:KK
            for ii = 1:nfore
                y_t_other(ii,:,nmod,irep-t0) = y_fore(ii,:,nmod);

                MAFE_other(irep-t0,:,ii,nmod) = abs(Yraw_f(irep+ii,:) - squeeze(y_t_other(ii,:,nmod,irep-t0)));
                MSFE_other(irep-t0,:,ii,nmod) = (Yraw_f(irep+ii,:) - squeeze(y_t_other(ii,:,nmod,irep-t0))).^2; 
            end
            PL_other(irep-t0,:,nmod) = PL(:,nmod);
        end
    end
end
%======================== END KALMAN FILTER ESTIMATION ========================
clc;
toc;

model = 'other_FAVAR';
save(sprintf('%s_%g_%g_%g_%g.mat',model,nlag,l_1,l_2,l_4),'-mat');
