function y=linls_direct(A,data,tau,b,eps)
% solves 
% min 1/2 || Ax - data ||_2^2 + tau*||x-b||_2^2
% with normal equation
% (A'*A + tau) x = A'*data + tau*b;

y = (A'*A + tau) \ (A'*data + tau*b);        