% function [fac,lam]=extract(data,k) extracts first k principal components from
% t*n matrix data, loadings are normalized so that lam'lam/n=I, fac is t*k, lam is n*k
function [fac,lam]=extract(data,k)
[t,n]=size(data);
xx=data'*data;
[evec,eval]=eig(xx);

% sorting evec so that they correspond to eval in descending order
[eval,index]=sort(diag(eval));
index=flipud(index); 		   % to get descending order
evc=zeros(n,n);
for i=1:n
   evc(:,i)=evec(:,index(i));
end

lam = sqrt(n)*evc(:,1:k);
fac=data*lam/n;
