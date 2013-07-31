close all
clear all


N=64;
x0=phantom(N);
D=rand(N,N)>.5;
D(1,1)=1;
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

toVec=@(x)reshape(x,[N*N 1]);
toImg=@(x)reshape(x,[N N]);

% fun =@(tau,b)lsqnonlin(@(x)DirectProjection(x,A,data,tau,b),x0(:),LowerBound(:),UpperBound(:),options);
% fun2=@(tau,b)lsqnonlin(@(x)ForwardProjection(x,data,tau,b),x0(:),LowerBound(:),UpperBound(:),options);

% use LBFGS for linear operator A, where A' can be computed
fun3=@(tau,b,eps,x0)toImg(LBFGS(toVec(x0),@(y)fun_grad(y,A,data,tau,b),eps,10,50));

% Use closed form solution for minimization
fun4=@(tau,b,eps,x0)(A'*A + tau) \ (A'*data + tau*b);

% only function evaluation needed here. Super slow:
J=@(x)toVec(A*toImg(x)-data);
fun5=@(tau,b,eps,x0)toImg(LBFGS(toVec(x0),@(y)fun_grad_FD(y,J,tau,toVec(b)),eps,20,20));

% use approximated gradient using random projections
R=rand(N*N);
[Q R]=qr(R);
fun6=@(tau,b,eps,x0)toImg(LBFGS(toVec(x0),@(y)fun_grad_FD_rand(y,J,tau,toVec(b),Q(:,1:(N*N))),eps,10,50));


%%
la=1e-4;
x=AugmentedLagrangian(fun5,[N N],struct('verbose',10,'tau',2*la,'lambda',la,'l2eps',1e-6));
