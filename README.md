Augmented Lagranian Solver 
=====================

     min (y,x)  1/2 ||f(y)||_2^2 + lambda Phi(x) s.t. y=x

using an augmented lagrangian approach with alternating direction 

the lagrangian for this subproblem is of the form

 L(y,x,Multipler,lambda) =
      1/2 ||f(y)||_2^2 + lambda Phi(x) + tau/2 ||y-x-Multiplier/tau||_2^2

L2 subproblem:
--------------
fun is a function handle for the L2 subproblem and has the form
function y=fun(tau,b,eps)
and should solve the L2 subproblem

min y    1/2 ||f(y)||_2^2 + tau/2 || y - b ||_2^2 with accuracy eps 

L1 subproblem:
--------------
the solution to the L1 subproblem depends on the regularizer, Phi(x), but
is of the form

min x    Phi(x) + tau/2 || x - b ||_2^2 
