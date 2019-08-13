% FAVAR_PC_DOZ - Forecasts from FAVAR models (constant parameters everywhere)
%                estimated using principal components (PC) and OLS, and the
%                two-step method of Doz, Giannone, Reichlin (2011).
%-----------------------------------------------------------------------------------------
% The model is:
%     _    _     _          _     _    _     _    _
%    | y[t] |   |   I    0   |   | y[t] |   |   0  |
%    |      | = |            | x |      | + |      |
%	 | x[t] |   |  Ly    Lf  |   | f[t] |   | e[t] |
%     -    -     -          -     -    -     -    -
%	 
%     _    _         _      _
%    | y[t] |       | y[t-1] |   
%    |      | = B x |        | + u[t]
%    | f[t] |       | f[t-1] |   
%     -    -         -      -     
% where L = (Ly ; Lf) and B are coefficients, f[t] are factors, e[t]~N(0,V)
% and u[t]~N(0,Q).
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
t0 = '1990Q1';    % Set initial estimation period
              
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

% =========================| PRIORS |================================
% Initial condition on the factors
factor_0.mean = zeros(k,1);
factor_0.var = 4*eye(k);

% Initialize matrix of forecasts
f_l = zeros(t,2);
y_fore = zeros(nfore,p,2);
y_t_FAVAR = zeros(nfore,p,2,t-t0);
Yraw_f = [ydata ; NaN(nfore,p)];
PL = zeros(p,2);

MAFE_PC = zeros(t-t0,p,nfore);
MSFE_PC = zeros(t-t0,p,nfore);
PL_PC = zeros(t-t0,p);

MAFE_DOZ = zeros(t-t0,p,nfore);
MSFE_DOZ = zeros(t-t0,p,nfore);
PL_DOZ = zeros(t-t0,p);

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
 
    % Estimate Principal Components and obtain OLS estimates of parameters
    X = X_st;
    [F,L] = extract(X,nfac);
    F(isnan(F))=0;
    YX = [Y,X];
    YF_PC = [Y,F];
    [L_OLS,B_OLS,beta_OLS,SIGMA_OLS,Q_OLS] = ols_pc_dfm(YX,YF_PC,L,y_true,n,p,r,nfac,nlag);
    
    % Estimate the Kalman Filter facotr of Doz et al.
    B_DOZ = [beta_OLS'; eye(r*(nlag-1)) zeros(r*(nlag-1),r)];
    Q_DOZ = [Q_OLS zeros(r,r*(nlag-1)); zeros(r*(nlag-1),k)];
    [Fdraw] = Kalman_companion(YX,0*ones(k,1),10*eye(k),L_OLS,(SIGMA_OLS + 1e-10*eye(q)),B_DOZ,Q_DOZ);
    YX = [Y,X];
    YF_DOZ = [Y,Fdraw(:,r)];    
    
    % Create lags on the (FA) VAR part
    YY_PC = YF_PC(nlag+1:end,:); XX_PC = mlag2(YF_PC,nlag); XX_PC = XX_PC(nlag+1:end,:);  
    YY_DOZ = YF_PC(nlag+1:end,:); XX_DOZ = mlag2(YF_DOZ,nlag); XX_DOZ = XX_DOZ(nlag+1:end,:);
    
    %=========
    if irep>t0 && irep<t
        factors_pc = [YF_PC(end,:) XX_PC(end,1:end-r)]';
        factors_doz = [YF_DOZ(end,:) XX_DOZ(end,1:end-r)]';
        [y_f_pc,PL_pc] = DMAFCI_fore(ydata(irep+1,:),B_DOZ,factors_pc,Q_DOZ,p,nfore);
        [y_f_doz,PL_doz] = DMAFCI_fore(ydata(irep+1,:),B_DOZ,factors_doz,Q_DOZ,p,nfore);
        y_fore(:,:,1) = y_f_pc;
        y_fore(:,:,2) = y_f_doz;
    end    
    
    for ii = 1:nfore
        y_t_FAVAR(ii,:,1,irep-t0) = y_fore(ii,:,1);
        y_t_FAVAR(ii,:,2,irep-t0) = y_fore(ii,:,2);
            
        MAFE_PC(irep-t0,:,ii) = abs(Yraw_f(irep+ii,:) - squeeze(y_t_FAVAR(ii,:,1,irep-t0)));
        MAFE_DOZ(irep-t0,:,ii) = abs(Yraw_f(irep+ii,:) - squeeze(y_t_FAVAR(ii,:,2,irep-t0)));
            
        MSFE_PC(irep-t0,:,ii) = (Yraw_f(irep+ii,:) - squeeze(y_t_FAVAR(ii,:,1,irep-t0))).^2; 
        MSFE_DOZ(irep-t0,:,ii) = (Yraw_f(irep+ii,:) - squeeze(y_t_FAVAR(ii,:,2,irep-t0))).^2; 
    end
    PL_PC(irep-t0,:) = PL_pc;
    PL_DOZ(irep-t0,:) = PL_doz;
end
%======================== END KALMAN FILTER ESTIMATION ========================
clc;
toc;

model = 'FAVAR_PC_DOZ';
save(sprintf('%s_%g.mat',model,nlag),'-mat');

% matlabpool close
