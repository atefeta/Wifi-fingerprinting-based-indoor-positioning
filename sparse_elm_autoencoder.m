function x = sparse_elm_autoencoder(a1,h1,lam,itrs)
%a1=A1;h1=H1;lam=1e-3;itrs=50
AA = (a1') * a1;
eiga1=eig(AA);
Lf = max(eig(AA));
Li = 1/Lf;
alp = lam * Li;
m = size(a1,2);
n = size(h1,2);
x = zeros(m,n);
yk = x;
tk = 1;
L1 = 2 * Li * AA;
L2 = 2 * Li * a1' * h1;
% tic
for i = 1:itrs
  ck = yk - L1*yk + L2;
  x1 = (max(abs(ck)-alp,0)).*sign(ck);
  tk1 = 0.5 + 0.5*sqrt(1+4*tk^2);
  tt = (tk-1)/tk1;
  yk = x1 + tt*(x-x1);
  tk = tk1;
  x = x1;
end
% toc

