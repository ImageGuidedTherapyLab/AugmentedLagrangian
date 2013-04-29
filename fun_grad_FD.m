function [y,g] = fun_grad_FD(x,fun,tau,b)
%% evaluates the objective
% f(x) = 1/2 || F(x) ||_2^2 + tau*||x-b||_2^2
% and \nabla f(x) using FD
% @wstefan
h=1e-8;
n=length(x);
d=fun(x);
y = 1/2*(d'*d)+tau/2*(x-b)'*(x-b);
if nargout > 1
    g=zeros(n,1);
    parfor i=1:length(x)
        e=zeros(n,1);
        e(i)=1;
        g(i) = (fun(x+h*e)-d)'*d/h + tau*(x(i)-b(i));
    end
end