function [beta_reshape] = reshape_tvp(theta_update,p,K,M)
% Reshape column of TVP-VAR coefficients in matrix form

bbtemp = theta_update(M+1:K,:);  % get the draw of B(t) at time i=1,...,T  (exclude intercept)                   
splace = 0;
biga = 0;
for ii = 1:p                                               
    for iii = 1:M                   
        biga(iii,(ii-1)*M+1:ii*M) = bbtemp(splace+1:splace+M,1);
        splace = splace + M;
    end
end
beta_reshape = [theta_update(1:M,:)' ;biga'];