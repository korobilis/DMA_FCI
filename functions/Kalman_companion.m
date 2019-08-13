function [Sdraw] = Kalman_companion(data,S0,P0,H,R,F,Q)

[t,nm]=size(data);
kml=size(S0,1);
km=size(H,2);
% KF
Sp=S0;  % p for prediction: S(t|t-1), Stt denotes S(t|t)
Pp=P0;
S=zeros(t,kml);
P=zeros(kml^2,t);
for i=1:t
    y = data(i,:)';
    nu = y - H*Sp(1:km);   % conditional forecast error
    f = H*Pp(1:km,1:km)*H' + R;    % variance of the conditional forecast error
    finv=H'/f;    
    Stt = Sp + Pp(:,1:km)*finv*nu;
    Ptt = Pp - Pp(:,1:km)*finv*(H*Pp(1:km,:));    
    if i < t
        Sp = F*Stt;
        Pp = F*Ptt*F' + Q;
    end    
    S(i,:) = Stt';
    P(:,i) = reshape(Ptt,kml^2,1);
end

% draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
Sdraw=zeros(t,kml);
Sdraw(t,:)=S(t,:);

% iterate 'down', drawing at each step, use modification for singular Q
Qstar=Q(1:km,1:km);
Fstar=F(1:km,:);

for i=1:t-1
    Sf = Sdraw(t-i+1,1:km)';
    Stt = S(t-i,:)';
    Ptt = reshape(P(:,t-i),kml,kml);
    f = Fstar*Ptt*Fstar' + Qstar;
    finv = Fstar'/f;
    nu = Sf - Fstar*Stt;    
    Smean = Stt + Ptt*finv*nu;
    Svar = Ptt - Ptt*finv*(Fstar*Ptt);    
    Sdraw(t-i,:) = Smean';
end
Sdraw=Sdraw(:,1:km);