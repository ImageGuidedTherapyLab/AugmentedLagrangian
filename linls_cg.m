function y=linls_cg(A,At,data,tau,b,eps,maxiter)
% solves 
% min 1/2 || Ax - data ||_2^2 + tau*||x-b||_2^2
% with normal equation
% (A'*A + tau) x = A'*data + tau*b;

[N M]=size(b);

rhs=reshape(At(data) + tau*b,[N*M 1]);
lhs=@(x) reshape(At(A(reshape(x,[N M]))),[N*M 1])+tau*x;
y = reshape(pcg(lhs,rhs,eps,maxiter),[N M]);
