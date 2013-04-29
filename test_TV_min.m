close all
clear all


N=256;
x0=phantom(N);
D=rand(N,N)>.5;
A=Convolution(ifft2(D));
%A=Convolution.Gauss([N N],.01);

data=A*x0+0*complex(randn(N,N),randn(N,N));

fun =@(tau,b,eps)linls_direct(A,data,tau,b,eps);
fun2=@(tau,b,eps)linls_cg(@(x)A*x,@(x)A'*x,data,tau,b,0);

x1=fun2(0.01,zeros([N N]),0);


%%
la=1e-4;
x=TV_min(fun2,[N N],struct('verbose',10,'tau',2*la,'lambda',la,'l2eps',1e-6));
