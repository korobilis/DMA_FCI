% TVP_FAVAR - Time-varying parameters factor-augmented VAR using an adaptive Kalman filter
% with EWMA filter covariance estimation
% MULTIPLE MODEL CASE (DYNAMIC MODEL AVERAGING - DMA)
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
k = nlag*r;      % number of sampled factors

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
[b_prior,Vb_prior] = Minn_prior_KOOP(4,r,nlag,m); % Obtain a Minnesota-type prior
beta_0.mean = b_prior;
beta_0.var = Vb_prior;
% Initial condition on the covariance matrices
V_0 = .1*eye(q); V_0(1:p,1:p) = 0;
Q_0 = .1*eye(r);

% Put all decay/forgetting factors together in a vector
l = [l_1; l_2; l_3; l_4];

% Initialize matrix of forecasts
f_l = zeros(t,KK);
factors_all = zeros(t,KK);

%======================= BEGIN KALMAN FILTER ESTIMATION =======================
tic;
       
X_st = standardize_miss(xdata(1:t,:));% + 1e-20;         
X_st(isnan(X_st)) = 0;               
%[Y,Ymeans,Ystds] = standardize2(ydata(1:t,:));
Y = ydata(1:t,:);   
% Iterate over all models   
for nmod = 1:KK
    if mod(nmod,ceil(KK./1000)) == 0
        disp([num2str(100*(nmod/KK)) '% completed'])       
        toc;   
    end
    % Extract Principal Components using data up to time irep
    X = X_st.*repmat(index{nmod,1},size(X_st,1),1);
    [F,Lf] = extract(X,nfac);
    YX = [Y,X];
    YF = [Y,F];
        
    % Estimate the FCI using the method in Koop and Korobilis (2013):
    % ====| STEP 1: Update Parameters Conditional on PC 
    [beta_t,beta_new,lambda_t,V_t,Q_t] = KFS_parameters(YX,YF,l,nfac,nlag,y_true,k,m,p,q,r,t,lambda_0,beta_0,V_0,Q_0);
    % ====| STEP 1: Update Factors Conditional on TV-Parameters   
    [factor_new,Sf_t_new] = KFS_factors(YX,lambda_t,beta_t,V_t,Q_t,nlag,k,r,q,t,factor_0);
    
    % Identify the sign of the factor so that it is an FCI: if the factor is not negative in 2008Q4
    % (sort of a "peak of the financial crisis"), then multiply the factor by -1. 
    if factor_new(r,strcmp(yearlab,'2008Q4')==1)>0
        factor_new(r,:) = -factor_new(r,:);
    end    
    
    % Save the factor estimate of model nmod    
    factors_all(:,nmod) = factor_new(r,:)';
    
    for irep = 1:t
        % Obtain Predictive Likelihood of the macro variables (f_l)
        YFn = [Y(1:irep,:), factor_new(r,1:irep)'];       
        YY = YFn(nlag+1:end,:); XX = mlag2(YFn,nlag); XX = XX(nlag+1:end,:);
        if irep > nlag + 1
            mm = XX(end,:)*beta_t(1:r,1:end,irep)';
            % Obtain the "predictive" likelihood of the p macro variables
            f_l(irep,nmod) = mvnpdfs(YFn(irep,2),mm(2),Q_t(2,2,irep));
        else  % for the first few observations (for which we cannot take lags) use the likelihood of the factor equation
            f_l(irep,nmod) = mvnpdfs(X(irep,:),YFn(end,:)*lambda_t(p+1:end,:,irep)',V_t(p+1:end,p+1:end,irep));
        end
    end
end
for irep = 1:t
    % Get sum of probabilities for all models
    if irep > 1
        sum_prob_omega = sum((omega_update(irep-1,:)).^alpha);  % this is the sum of the nom model probabilities (all in the power of the forgetting factor 'eta')
    end
    for nmod = 1:KK
        % DMA predict step
        if irep > 1
            omega_predict(irep,nmod) = (omega_update(irep-1,nmod).^alpha)./(sum_prob_omega);       
        end
        % Calculate weights w_t (used below for DMA update step)
        w_t(irep,nmod) = omega_predict(irep,nmod)*f_l(irep,nmod); 
    end
    
    % Convert DMA weights to DMA probabilities (DMA update step)
    omega_update(irep,:) = (w_t(irep,:))./(sum(w_t(irep,:),2));
    [value_DMS,i_DMS] = max(omega_update(irep,:));   % This is maximum probability at time t (DMS)    
end
%======================== END KALMAN FILTER ESTIMATION ========================
clc;
toc;

% Calculate expected size of variables selected by DMA
pSize = zeros(KK,1);
eSize = zeros(t,1);
for ii=1:t
    for kk=1:KK        
        % Weighted size (pSize) = model probability x size of model
        % Note that size of each model in excess of the variables always included
        pSize(kk,:) = omega_update(ii,kk)*(sum(index{kk},2) - max(var_no_dma));
    end
    % Expected size is the sum of all weighted model sizes
    eSize(ii,:) = sum(pSize,1);
end

varnames(var_no_dma)=[];

% Some basic, inefficient code to obtain probability for each of the 16 variables
dma_vars = size(xdata,2) - length(var_no_dma);
prob_variable = zeros(dma_vars,t);
for irep = 1:dma_vars
    g=[];
    for ii=1:KK
        d = sum(index_temp2{ii,1}== length(var_no_dma)+irep);
        if d==1
            g = [g ; ii];
        end
    end
    prob_variable(irep,:) = sum(squeeze(omega_update(:,g))'); %#ok<UDIM>
end
load xdata.dat;
for i=1:20; term(i) = sum(isnan(xdata(:,i))); end; term(var_no_dma)=[];
for i=1:dma_vars; prob_variable(i,1:term(i)) = 0.5; end

% Plot DMA probabilities for each variable
figure      
ticks=0:48:floor(t./10)*10;       
ticklabels=yearlab(ticks+1); 
nplots = max(find(mod(dma_vars,1:5)==0));
if nplots == 1; nplots = 4; end
for i = 1:dma_vars
    subplot(ceil(dma_vars./(ceil(dma_vars./nplots))),ceil(dma_vars./nplots),i)
    pl1=plot(prob_variable(i,:),'r');
    set(gca,'XTick',ticks)
    set(gca,'XTickLabel',ticklabels)
    set(pl1,'LineWidth',2)
    xlim([1 t])
    ylim([0 1])
    grid on        
    title(['Probability of ' cell2mat(varnames(i))])
end

% Plot expected size of variables included in DMA
figure
pl2=plot(eSize,'r');
set(gca,'XTick',ticks)
set(gca,'XTickLabel',ticklabels)
set(pl2,'LineWidth',2)
xlim([1 t])
ylim([0 dma_vars])
grid on   
title('Expected model size in DMA')

model = 'DMA_probs';
save(sprintf('%s_%g_%g_%g_%g_%g_%g.mat',model,nlag,alpha,l_1,l_2,l_3,l_4),'-mat');

% Plot the DMA factor from the chosen model
factor_DMA = sum(factors_all.*omega_update,2);
figure
pl3=plot(factor_DMA,'r');
set(gca,'XTick',ticks)
set(gca,'XTickLabel',ticklabels)
set(pl3,'LineWidth',2)
xlim([1 t])
grid on   
title('FCI implied by DMA')

% matlabpool close