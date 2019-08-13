
function [L,bb,beta_OLS,sigma2,sigmaf] = ols_pc_dfm(YX,YF,Lf,y_true,n,p,r,nfac,nlag) 

%Lf = Lf./Lf(1,1);
t=size(YX,1);
% Obtain L (the loadings matrix)
if y_true == 1
    L = (olssvd(YX,YF))';
elseif y_true == 0
    L = [eye(p) zeros(p,nfac) ;zeros(n,p) Lf];
end
% Obtain the errors of the factor equation
e = YX - YF*L';
sigma2 = diag(diag(e'*e./t));
% Obtain the errors of the VAR(1) equation
yy = YF(nlag+1:end,:);
xx = mlag2(YF,nlag); xx = xx(nlag+1:end,:);
beta_OLS = inv(xx'*xx)*(xx'*yy);
sigmaf = (yy - xx*beta_OLS)'*(yy - xx*beta_OLS)/(t-nlag-1);
beta_var = kron(sigmaf,inv(xx'*xx));
bb = [];
for i = 1:nlag
    g = beta_OLS((i-1)*r+1:i*r,1:r);
    bb = [bb ; g(:)];
end