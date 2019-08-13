% DMA_TVP_FAVAR - Time-varying parameters factor-augmented VAR using an adaptive Kalman filter
% with EWMA filter covariance estimation 
% MULTIPLE MODEL CASE (DYNAMIC MODEL AVERAGING - DMA): NONINFORMATIVE PRIOR
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
%  - This code does DMA on the loadings matrix L[f,t]
%  - This code uses NONINFORMATIVE PRIORS AND INITIAL CONDITIONS
%  - This version uses the Parallel Computing Toolbox (if you do not have
%  this installed, please contact me at Dimitris.Korobilis@glasgow.ac.uk
%-----------------------------------------------------------------------------------------
% Written by Dimitris Korobilis
% University of Glasgow
% This version: 08 July, 2013
%-----------------------------------------------------------------------------------------

clear all;
close all;
clc;

% % Distribute the job in 12 workers (two six-core Xeon processors), using local configuration (single
% % machine). For even better performance you can convert the code to batch mode and load it on a cluster.
% if matlabpool('size') < 2
%     matlabpool open 12
% end

% Add path of data and functions
addpath('data');
addpath('functions');

%-------------------------------USER INPUT--------------------------------------
% Model specification
nfac = 1;         % number of factors
nlag = 4;         % number of lags of factors

% Forgetting factor for DMA
alpha = 0.99;

% Control the amount of variation in the measurement and error variances
l_1 = 0.96;       % Decay factor for measurement error variance
l_2 = 0.96;       % Decay factor for factor error variance
l_3 = 0.99;       % Forgetting factor for loadings error variance
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
                  
% Choose how to do DMA
var_no_dma = 1;   % Choose variables always included in each model (to help identify factors).
                  % Single model case is var_no_dma = 1:20 (all variables included, no dma).                  
                  
% Forecasting
nfore = 4;        % Forecast horizon (note: forecasts are iterated)
t0 = '1990Q1';    % Set initial estimation period  
              
%----------------------------------LOAD DATA----------------------------------------
% Load Koop and Korobilis (2012) quarterly data
% load data used to extract factors
load xdata.dat;
% load data on inflation, gdp and the interest rate 
load ydata.dat;    % first log differences (inflation/gpd)
load ydata2.dat;   % second log differences (inflation/gpd)
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

% Choose variables always included in each model and short them first
xtemp = xdata(:,var_no_dma);
xdata(:,var_no_dma)=[];
xdata = [xtemp xdata];

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
NN = size(xdata,2) - size(xtemp,2);
comb = cell(NN,1);
for nn = 1:NN
    % 'comb' has NN cells with all possible combinations for each NN
    comb{nn,1} = combntns(1:NN,nn);
end
KK = (2.^NN);
index_temp = cell(KK,1);
dim = zeros(1,NN+1);
for nn=1:NN
    dim(:,nn+1) = size(comb{nn,1},1);
    for jj=1:size(comb{nn,1},1)
        % Take all possible combinations from variable 'comb' and sort them
        % in each row of 'index'. Index now has a vector in each K row that
        % indexes the variables to be used, i.e. for N==3:
        % index = {[1] ; [2] ; [3] ; [1 2] ; [1 3] ; [2 3] ; [1 2 3]}
        index_temp{jj + sum(dim(:,1:nn)),1} = comb{nn,1}(jj,:);
    end
end

index_temp2 = cell(KK,1);
% Fix now the dimensions of the "index" variable to include the indexes for
% the variables that are not subject to DMA
for iii = 1:KK  %#ok<*BDSCI>
    index_temp2{iii,1} = index_temp{iii,1} + size(xtemp,2);
end

index = cell(KK,1);
for iii=1:KK
    index{iii,1} = zeros(1,n);
    index{iii,1}(1,[1:size(xtemp,2) index_temp2{iii,1}]) = 1;
end

% =========================| PRIORS |================================
% Use common prior settings for all models (no need to create KK-dimensional arrays)
omega_predict = (1/KK)*ones(t,KK);
omega_update = (1/KK)*ones(t,KK);
w_t = zeros(t,KK);
% Initial condition on the factors
factor_0.mean = zeros(k,1);
factor_0.var = 4*eye(k);
% Initial condition on lambda_t
lambda_0.mean = zeros(q,r);
lambda_0.var = 4*eye(r);
% Initial condition on beta_t
[b_prior,Vb_prior] = Minn_prior_KOOP(0.1,r,nlag,m); % Obtain a Minnesota-type prior
beta_0.mean = b_prior;
beta_0.var = Vb_prior;
% Initial condition on the covariance matrices
V_0 = 1*eye(q); V_0(1:p,1:p) = 0;
Q_0 = 1*eye(r);

% Put all decay/forgetting factors together in a vector
l = [l_1; l_2; l_3; l_4];

% Initialize matrix of forecasts
f_l = zeros(t,KK);
y_fore = zeros(nfore,p,KK);
y_t_DMA = zeros(nfore,p,t-t0);
y_t_DMS = zeros(nfore,p,t-t0);
Yraw_f = [ydata ; NaN(nfore,p)];
PL = zeros(p,KK);

MAFE_DMA = zeros(t-t0,p,nfore);
MAFE_DMS = zeros(t-t0,p,nfore);
MSFE_DMA = zeros(t-t0,p,nfore);
MSFE_DMS = zeros(t-t0,p,nfore);
PL_DMA = zeros(t-t0,p);
PL_DMS = zeros(t-t0,p);

%======================= BEGIN KALMAN FILTER ESTIMATION =======================
tic;

for irep = t0+1:t-1
    %if mod(irep,ceil(t./40)) == 0
        disp([num2str(100*((irep-t0)/(t-t0-1))) '% completed'])       
        toc;   
    %end
    % Standardize data up to time irep
    if irep<=t0
        X_st = standardize_miss(xdata(1:t0,:));% + 1e-20;   
        X_st(isnan(X_st)) = 0;               
        %[Y,Ymeans,Ystds] = standardize2(ydata(1:t0,:));
        Y = ydata(1:t0,:);
    elseif irep>t0      
        X_st = standardize_miss(xdata(1:irep,:));% + 1e-20;
        X_st(isnan(X_st)) = 0;
        %[Y,Ymeans,Ystds] = standardize2(ydata(1:irep,:));   
        Y = ydata(1:irep,:);
    end
        
    % Get sum of probabilities for all models
    if irep > 1
        sum_prob_omega = sum((omega_update(irep-1,:)).^alpha);  % this is the sum of the nom model probabilities (all in the power of the forgetting factor 'eta')
    end
    
    % Iterate over all models
    parfor nmod = 1:KK %#ok<PFUIX>
        % Extract Principal Components using data up to time irep
        X = X_st.*repmat(index{nmod,1},size(X_st,1),1);
        [F,Lf] = extract(X,nfac);
        YX = [Y,X];
        YF = [Y,F];
        
        % Estimate the FCI using the method in Koop and Korobilis (2013):
        % ====| STEP 1: Update Parameters Conditional on PC 
        [beta_t,beta_new,lambda_t,V_t,Q_t] = KFS_parameters(YX,YF,l,nfac,nlag,y_true,k,m,p,q,r,irep,lambda_0,beta_0,V_0,Q_0);
        % ====| STEP 1: Update Factors Conditional on TV-Parameters   
        [factor_new,Sf_t_new] = KFS_factors(YX,lambda_t,beta_t,V_t,Q_t,nlag,k,r,q,irep,factor_0);
        
        % Obtain Predictive Likelihood of the 3 macro variables (f_l)
        YFn = [Y,factor_new(r,:)'];
        YY = YFn(nlag+1:end,:); XX = mlag2(YFn,nlag); XX = XX(nlag+1:end,:);
        if irep > nlag + 1
            mm = XX(end,:)*beta_t(1:r,1:end,irep)';
            f_l(irep,nmod) = mvnpdfs(YFn(end,1:p),mm(1:p),Q_t(1:p,1:p,irep));
        else
            f_l(irep,nmod) = mvnpdfs(X(irep,:),YFn(end,:)*lambda_t(p+1:end,:,irep)',V_t(p+1:end,p+1:end,irep));
        end
        % DMA predict step
        if irep > 1
            omega_predict(irep,nmod) = (omega_update(irep-1,nmod).^alpha + 1e-30)./(sum_prob_omega + 1e-30);
        end
        % Calculate weights w_t (used below for DMA update step)
        w_t(irep,nmod) = omega_predict(irep,nmod)*f_l(irep,nmod);
        %=========
        if irep>t0 && irep<t
            % 1/ Forecast of model nmod, period irep
            factors = [YFn(end,:) XX(end,1:end-r)]';
            [y_f,PL_1] = DMAFCI_fore(ydata(irep+1,:),beta_t(:,:,irep),factors,Q_t(:,:,irep),p,nfore);
            y_fore(:,:,nmod) = y_f;
            PL(:,nmod) = PL_1;
        end        
    end
    
    % Convert DMA weights to DMA probabilities (DMA update step)
    omega_update(irep,:) = (w_t(irep,:))./(sum(w_t(irep,:),2));
    [value_DMS,i_DMS] = max(omega_update(irep,:));   % This is maximum probability at time t (DMS)
    
    if irep>t0
        for ii = 1:nfore
            weight_pred = zeros(1,p);
            for nmod = 1:KK
                temp_predict = y_fore(ii,:,nmod).*omega_update(irep,nmod);                
                weight_pred = weight_pred + temp_predict;
            end            
                        
            y_t_DMA(ii,:,irep-t0) = mean(weight_pred,3);
            y_t_DMS(ii,:,irep-t0) = y_fore(ii,:,i_DMS);
            
            MAFE_DMA(irep-t0,:,ii) = abs(Yraw_f(irep+ii,:) - squeeze(y_t_DMA(ii,:,irep-t0)));
            MSFE_DMA(irep-t0,:,ii) = (Yraw_f(irep+ii,:) - squeeze(y_t_DMA(ii,:,irep-t0))).^2; 
            
            MAFE_DMS(irep-t0,:,ii) = abs(Yraw_f(irep+ii,:) - squeeze(y_t_DMS(ii,:,irep-t0)));
            MSFE_DMS(irep-t0,:,ii) = (Yraw_f(irep+ii,:) - squeeze(y_t_DMS(ii,:,irep-t0))).^2; 
        end

        PL_DMA(irep-t0,:) = sum(PL.*repmat(omega_update(irep,:),p,1),2);
        PL_DMS(irep-t0,:) = PL(:,i_DMS);
    end
end
%======================== END KALMAN FILTER ESTIMATION ========================
clc;
toc;

model = 'DMA_FAVAR';
save(sprintf('%s_%g_%g_%g_%g_%g_%g.mat',model,nlag,alpha,l_1,l_2,l_3,l_4),'-mat');

% matlabpool close
