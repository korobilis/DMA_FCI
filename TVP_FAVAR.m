% TVP_FAVAR - Time-varying parameters factor-augmented VAR using EWMA Kalman filters 
% SINGLE MODEL CASE
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
%  - This code estimates a single model
%-----------------------------------------------------------------------------------------
% Written by Dimitris Korobilis
% University of Glasgow
% This version: 04 July, 2012
%-----------------------------------------------------------------------------------------

clear all; close all;
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
l_3 = 0.99;       % Decay factor for loadings error variance
l_4 = 0.99;       % Decay factor for VAR coefficients error variance

% Select if y[t] should be included in the measurement equation (if it is
% NOT included, then the coefficient/loading L[y,t] is zero for all periods
y_true = 1;       % 1: Include y[t]; 0: Do not include y[t]

% Select data set to use
select_data = 2;  % 1: 81 variables; 2: 20 variables (as in DMA)

% Select subsample (NOTE: only if select_data = 1)
sample = 2;       % 1: Use balanced panel of 18 variables after 1970:Q1
                  % 2: Use all data from 1959:Q1 (unbalanced)
                  % 3: Use all data from 1980:Q1 (unbalanced)

% Select transformations of the macro variables in Y
transf = 1;       % 1: Use first (log) differences (only for CPI, GDP, M1)
                  % 2: Use annualized (CPI & GDP) & second (log) differences (M1)    
                                    
% Select a subset of the 6 variables in Y                  
subset = 1;       % 1: Infl.- GDP - Int. Rate (3 vars)
                  % 2: Infl. - Unempl. - Int. Rate (3 vars)
                  % 3: Infl. - Inf. Exp. - GDP - Int. Rate (4 vars)
                  % 4: Infl. - Inf. Exp. - GDP - M1 - Int. Rate (5 vars)
                  % 5: Infl. - Inf. Exp. - Unempl. - M1 - Int. Rate (5 vars)

% Impulse responses
nhor = 21;        % Impulse response horizon
resp_int = 4;     % Chose the equation # where the shock is to be imposed
shock_type = 1;   % 1: Cholesky - no dof adjustment 
                  % 2: Residual - 1 unit (0.5 increase of interest rate)      

% Forecasting
nfore = 16;        % Forecast horizon (note: forecasts are iterated)

% Plot graphs
plot_est = 1;     % 1: Plot graphs of estimated factors, volatilities etc; 0: No graphs
plot_imp = 1;     % 1: Plot graphs of impulse responses; 0: No grpahs
plot_fore = 1;    % 1: Plot graphs of impulse responses; 0: No grpahs

%----------------------------------LOAD DATA----------------------------------------
% Load Koop and Korobilis (2012) quarterly data
% load data used to extract factors
load xdata_all.dat;
load xdata.dat;
% load data on inflation, gdp and the interest rate 
load ydata.dat;
load ydata2.dat;
% load transformation codes (see file transx.m)
load tcode.dat;
% load the file with the dates of the data (quarters)
load yearlab.mat;


if transf == 2
    ydata = ydata2;
end

if select_data == 1
    xdata = xdata_all;
    % load the file with the names of the variables 
    load xnames.mat;   
    if sample==1
        xdata = xdata(46:end,[3:4 9:11 14 16 22 31:34 37:38 40:41 63 66]);   
        ydata = ydata(46:end,:);
        yearlab = yearlab(46:end);
        namesXY = ['Inflation' ; 'Infl. Exp.'; 'GDP'; 'Unemployemnt'; 'M1'; 'FedFunds'; varnames([3:4 9:11 14 16 22 31:34 37:38 40:41 63 66]) ];
    elseif sample == 2
        namesXY = ['Inflation' ; 'Infl. Exp.'; 'GDP'; 'Unemployemnt'; 'M1'; 'FedFunds'; varnames ];
    elseif sample == 3
        xdata = xdata(85:end,:);
        ydata = ydata(85:end,:);
        yearlab = yearlab(85:end);
        namesXY = ['Inflation' ; 'Infl. Exp.'; 'GDP'; 'Unemployemnt'; 'M1'; 'FedFunds'; varnames ];
    end
elseif select_data == 2
    % load the file with the names of the variables
    load varnames.mat;
    namesXY = ['Inflation' ; 'Infl. Exp.'; 'GDP'; 'Unemployemnt'; 'M1'; 'FedFunds'; varnames ];
end

% Demean and standardize data (needed to extract Principal Components)
xdata = standardize_miss(xdata) + 1e-10 ;
xdata(isnan(xdata)) = 0;
ydata = standardize(ydata);

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
end

% Define X and Y matrices
X = xdata;   % X contains the 'xdata' which are used to extract factors.
Y = ydata;   % Y contains inflation, gdp and the interest rate

% Set dimensions of useful quantities
t = size(Y,1); % t time series observations
n = size(X,2); % n series from which we extract factors
p = size(Y,2); % and p macro series
r = nfac + p;  % number of factors and macro series
q = n + p;     % number of observed and macro series
m = nlag*(r^2);  % number of VAR parameters
k = nlag*r;      % number of sampled factors

% Just a small check to avoid error in input
if resp_int > r
    disp(['Your VAR is of size ' mat2str(r) ' and you imposing a shock in equation ' mat2str(resp_int) '!!!'])
    error('Check again your input in "resp_int"')    
end
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

% Initialize impulse response analysis
impulses =  zeros(t,q,nhor);
if resp_int<=p
    scale = std(Y(:,resp_int));
else
    scale=1;
end
bigj = zeros(r,r*nlag);
bigj(1:r,1:r) = eye(r);

% Initialize matrix of forecasts
y_fore = zeros(nfore,r);
%----------------------------- END OF PRELIMINARIES ---------------------------
tic;
%======================= FAVAR ESTIMATION =======================
% Get PC estimate using xdata up to time t
X_st = standardize_miss(xdata(1:end,:));
X_st(isnan(X_st)) = 0;
[FPC2,LPC] = extract(X_st,nfac);
%Y = standardize(ydata(1:end,:));      
FPC = [Y, FPC2];  % macro data and FCI          
YX = [Y, X_st];   % macro data and financial data
[L_OLS,B_OLS,beta_OLS,SIGMA_OLS,Q_OLS] = ols_pc_dfm(YX,FPC,LPC,y_true,n,p,r,nfac,nlag);

% 1/ Estimate the FCI using Principal Component:
FPCA = FPC2;

% 2/ Estimate the FCI using the method by Doz, Giannone and Reichlin (2011):  
B_doz = [beta_OLS'; eye(r*(nlag-1)) zeros(r*(nlag-1),r)];
Q_doz = [Q_OLS zeros(r,r*(nlag-1)); zeros(r*(nlag-1),k)];  
[Fdraw] = Kalman_companion(YX,0*ones(k,1),10*eye(k),L_OLS,(SIGMA_OLS + 1e-10*eye(q)),B_doz,Q_doz);
FDOZ = Fdraw;  

% 3/ Estimate the FCI using the method in Koop and Korobilis (2013):
% ====| STEP 1: Update Parameters Conditional on PC 
[beta_t,beta_new,lambda_t,V_t,Q_t] = KFS_parameters(YX,FPC,l,nfac,nlag,y_true,k,m,p,q,r,t,lambda_0,beta_0,V_0,Q_0);
% ====| STEP 1: Update Factors Conditional on TV-Parameters   
[factor_new,Sf_t_new] = KFS_factors(YX,lambda_t,beta_t,V_t,Q_t,nlag,k,r,q,t,factor_0);

%======================== END FAVAR ESTIMATION ========================


%=================== IMPULSE RESPONSES AND FORECASTS ========================
for irep = 1:t
    % Assign coefficients
    bb = beta_new(:,irep);
    splace = 0; biga = 0;
    for ii = 1:nlag                                          
        for iii = 1:r           
            biga(iii,(ii-1)*r+1:ii*r) = bb(splace+1:splace+r,1)';       
            splace = splace + r;   
        end
    end
    B = [biga ; eye(r*(nlag-1)) zeros(r*(nlag-1),r)];
    beta_t(:,:,irep) = B;

    % =====================| Extract impulse responses |==========================
    % ============================================================================
    if shock_type == 2 
        shock = zeros(r,r);
        shock(resp_int,resp_int) = -1;
    elseif shock_type == 1
        % st dev matrix for structural VAR       
        shock = squeeze(Q_t(:,:,irep));                     
        shock = -chol(shock)';
    elseif shock_type ~= 1 && shock_type ~= 2 && shock_type ~= 3
        error('Wrong specification of shock')
    end

    % Now get impulse responses for 1 through nhor future periods
    impresp = zeros(r,r*nhor); % matrix to store initial response at each period
    impresp(1:r,1:r) = shock;  % First shock is the Cholesky of the VAR covariance
    bigai = beta_t(:,:,irep);
    for j = 1:nhor-1
        impresp(:,j*r+1:(j+1)*r) = bigj*bigai*bigj'*shock;
        bigai = bigai*beta_t(:,:,irep);
    end
    
    % Only for specified periods
    impf_m = zeros(r,nhor);
    jj=0;
    for ij = 1:nhor
        jj = jj + resp_int;    % restrict to the p-th equation, the interest rate
        impf_m(:,ij) = impresp(:,jj);
    end
    impulses(irep,:,:) = lambda_t(:,:,irep)*impf_m(:,:); % store draws of responses
    
    % ============================| Do forecasting |==============================
    % ============================================================================
    if irep==t
        for ii = 1:nfore
            FORECASTS = (beta_t(:,:,irep)^ii)*factor_new(:,irep);          
            y_fore(ii,:) = FORECASTS(1:r,:)';   
        end
    end    
end
%================== END IMPULSE RESPONSES AND FORECASTS ========================
toc;

clc;
disp('Nice, it worked!')
disp('You have estimated a Financial Conditions Index using a TVP-FAVAR')
disp('with time-varying loadings and stochastic volatility')
disp('                        ')
disp('The FAVAR coefficients from the PCA/Doz et al. (2011) method are in the matrices:')
disp('L (loadings), V (measurement covariance), B (FAVAR coefficients) and Q (state covariance)')
disp('The PCA factors are in the vector FPCA, the Doz et al. (2011) factors in FDOZ')
disp('                        ')
disp('The TVP-FAVAR coefficients can be found in the matrices:')
disp('lambda_t, V_t, beta_t and Q_t')


%===================| PLOT GRAPHS |=========================
% ----| PLOT ESTIMATES OF FACTORS AND VOLATILITIES
if plot_est==1
    % Plot factors
    if nfac==1
        ticks=0:30:floor(t./10)*10;
        ticklabels=yearlab(ticks+1);   
        figure
        plot([factor_new(r,:)' FDOZ(:,r) FPCA],'Linewidth',2)
        set(gca,'XTick',ticks)
        set(gca,'XTickLabel',ticklabels)
        xlim([1 t])
        grid on        
        title('Estimated Factors')
        legend('TVL-DFM','Doz et al. (2011)', 'static PCA')
    else
        ticks=0:30:floor(t./10)*10;
        ticklabels=yearlab(ticks+1);   
        figure
        plot(factor_update(1:nfac,:)','Linewidth',2)
        set(gca,'XTick',ticks)
        set(gca,'XTickLabel',ticklabels)
        xlim([1 t])
        grid on        
        title('TVL-DFM factors')
        figure
        plot(FDOZ,'Linewidth',2)
        set(gca,'XTick',ticks)
        set(gca,'XTickLabel',ticklabels)
        xlim([1 t])
        grid on        
        title('Doz et al. (2011) factors')   
        figure
        plot(FPCA,'Linewidth',2)
        set(gca,'XTick',ticks)
        set(gca,'XTickLabel',ticklabels)
        xlim([1 t])
        grid on       
        title('static PCA factors')
    end
    
    % Plot factor volatilities
    vols = [];
    f = cell(nfac,1);
    for ii = 1:nfac
        vols = [vols squeeze(Q_t(p+ii,p+ii,:))];
        f{ii,1} = ['volatility of the FCI'];
    end
    figure
    plot(vols,'Linewidth',2)
    set(gca,'XTick',ticks)
    set(gca,'XTickLabel',ticklabels)  
    xlim([1 t])
    grid on
    title('Volatility of the FCI')
    legend(f)

    % Plot (some) idiosyncratic volatilities
    volsV = [squeeze(V_t(p+2,p+2,:)) squeeze(V_t(p+4,p+4,:))];
    figure
    plot(volsV,'Linewidth',2)
    set(gca,'XTick',ticks)
    set(gca,'XTickLabel',ticklabels)
    xlim([1 t])    
    grid on    
    title(['Idiosyncratic volatilities of variable ' cell2mat(varnames(p+2)) ' and variable ' cell2mat(varnames(p+4))])
    legend(cell2mat(varnames(p+2)),cell2mat(varnames(p+4)))
end

% ----| PLOT IMPULSE RESPONSES AT EACH TIME PERIOD
if plot_imp == 1
    % Select variable to plot its impulse response:     
    selected_variable = 3; 
    ticks=0:12:floor(t./10)*10;
    ticklabels=yearlab(ticks+1);
    figure
    surf(squeeze(impulses(:,selected_variable,:))./scale);
    set(gca,'YTick',ticks)
    set(gca,'YTickLabel',ticklabels)
    set(gca,'XTick',0:3:nhor)
    ylim([1 t])
    xlim([1 nhor])
    title(['Impulse response of variable ' cell2mat(namesXY(selected_variable))])
    colormap('summer')
    colorbar
    camorbit(-15,0)
end

% ----| PLOT FORECASTS FOR EACH SERIES (SHADED AREA)
if plot_fore == 1
    figure
    ticks=0:24:floor(t./10)*10;
    ticklabels=yearlab(ticks+1);
    for iii = 1:r
        subplot(ceil(r./2),2,iii)       
        plot([factor_new(iii,:)' ; y_fore(:,iii)],'Linewidth',2)
        curax=axis;
        y=[curax(3) curax(4) curax(4) curax(3)];
        for i=1:length(t+1); 
            x=[t+1 t+1 t+nfore t+nfore]; 
            fill(x,y,'y');%[.8 .8 .8]);
        end;
        hold on;
        plot([factor_new(iii,:)' ; y_fore(:,iii)],'Linewidth',2)
        set(gca,'XTick',ticks)
        set(gca,'XTickLabel',ticklabels)
        xlim([1 t+nfore])
        grid on
        if iii == r
            title(['One step ahead forecasts of the FCI'])
        else
            title(['One step ahead forecasts of ' cell2mat(namesXY(iii))])
        end
    end
end
