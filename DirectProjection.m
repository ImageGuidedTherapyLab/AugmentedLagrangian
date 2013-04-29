function [residual,jacobian] = DirectProjection(y,A,data,tau,b)
% solve the L2 subproblem
%
% min y    1/2 ||f(y)||_2^2 + tau/2 || y - b ||_2^2 with accuracy eps 
%
% within the context of lsqnonlin solver, this problem is of the form
% of a concatenated vector for each L2 term
%
%                                         |    A y - data   |
% min y  || G(y) ||^2     with   G(y) =   | sqrt(tau) (y-b) |
%                                         

residual = [A*y - data ; sqrt(tau) *(y-b)];
jacobian = [A; sqrt(tau) * eye(size(y,1))];
