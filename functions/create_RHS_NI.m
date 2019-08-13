function [x_t,K] = create_RHS_NI(YY,M,p,t)

% No intercept case

K = p*(M^2); % K is the number of elements in the state vector

% Create x_t matrix.
% first find the zeros in matrix x_t
x_t = zeros((t-p)*M,K);
for i = 1:t-p
    ztemp = [];
    for j = 1:p
        xtemp = YY(i,(j-1)*M+1:j*M);
        xtemp = kron(eye(M),xtemp);
        ztemp = [ztemp xtemp];
    end
    x_t((i-1)*M+1:i*M,:) = ztemp;
end

