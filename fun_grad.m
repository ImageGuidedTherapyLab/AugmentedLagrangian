function [y,g] = fun_grad(x,A,data,tau,b)
%% evaluates the objective
% f(x) = 1/2 || Ax-data ||_2^2 + tau*||x-b||_2^2
% and \nabla f(x) using 
% (A'*A + tau)*x - A'*data - tau*b;
% @wstefan
sz=A.getSize();
d=reshape(A*reshape(x,sz)-data,[prod(sz) 1]);
xb=x-reshape(b,[prod(sz) 1]);
y = 1/2*(d'*d)+tau/2*xb'*xb;
if nargout > 1
    g=reshape((A'*A + tau)*reshape(x,sz) - A'*data - tau*b,[prod(sz) 1]);
end