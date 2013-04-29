close all
clear all


N=256;
x0=phantom(N);
D=rand(N,N)>.5;
A=Convolution(ifft2(D));
%A=Convolution.Gauss([N N],.01);

data=A*x0+0*complex(randn(N,N),randn(N,N));

eps = 1.e-8
options = optimset('jacobian','off','MaxIter',10000000000,'MaxFunEvals',10000000000,'TolFun',eps);
% L2 subproblem:
% --------------
% fun is a function handle for the L2 subproblem and has the form
% function y=fun(tau,b,eps)
% and should solve the L2 subproblem
%
% min y    1/2 ||f(y)||_2^2 + tau/2 || y - b ||_2^2 with accuracy eps 
%
% within the context of lsqnonlin solver, this problem is of the form
% of a concatenated vector for each L2 term
%
%                                         |       f(y)      |
% min y  || G(y) ||^2     with   G(y) =   | sqrt(tau) (y-b) |
%                                         
fun =@(tau,b)lsqnonlin(@(x)DirectProjection(x,A,data,tau,b),x0(:),LowerBound(:),UpperBound(:),options);
fun2=@(tau,b)lsqnonlin(@(x)ForwardProjection(x,data,tau,b),x0(:),LowerBound(:),UpperBound(:),options);

x1=fun2(0.01,zeros([N N]),0);


%%
la=1e-4;
x=AugmentedLagrangian(fun2,[N N],struct('verbose',10,'tau',2*la,'lambda',la,'l2eps',1e-6));
