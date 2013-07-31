function x=AugmentedLagrangian_group(fun,sz,opt)
% solves
% 
%      min (y,x)  1/2 ||f(y)||_2^2 + lambda Phi(x) s.t. y=x
% 
% using an augmented lagrangian approach with alternating direction 
% 
% the lagrangian for this subproblem is of the form
%
%  L(y,x,Multipler,lambda) =
%       1/2 ||f(y)||_2^2 + lambda Phi(x) + tau/2 ||y-x-Multiplier/tau||_2^2
% 
% L2 subproblem:
% --------------
% fun is a function handle for the L2 subproblem and has the form
% function y=fun(tau,b,eps,x0)
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
% x0 is the solution of the previous iteration and can be used as initial
% solution of the l2 solver.
%
%
% L1 subproblem:
% --------------
% the solution to the L1 subproblem depends on the regularizer, Phi(x), but
% is of the form
%
% min x    Phi(x) + tau/2 || x - b ||_2^2 
%
% sz is the problem size
% The L1 subproblem is solved as a group sparsity contraint. I.e. if x is
% zero anyware it is also zero along the third dimension. 
%
% opt.Nmax:  maximal number of iterations (default=100)
% opt.eps:   stopping criteria (default=1e-3)
% opt.sharp: sharpness parameter, larger is sharper but noisier. (default=6)

if not(exist('opt','var'))
    opt=struct();
end

Nmax   =get_opt(opt,'Nmax',100);
eps    =get_opt(opt,'eps',1e-3);
verbose=get_opt(opt,'verbose',1);
tau    =get_opt(opt,'tau',6);
reg    =get_opt(opt,'reg',3);
lambda =get_opt(opt,'lambda',1);
x0 =get_opt(opt,'x0',zeros(sz));

% set image size in x and y direction
Ny=sz(1); Nx=sz(2); M=sz(3);

% setup derivative operators for TK regularization.
Lx=Convolution.get_lx([Ny Nx]); 
Ly=Convolution.get_ly([Ny Nx]);

% setup auxiliary variables
Multiplier=zeros(Ny,Nx,M);
y=zeros(Ny,Nx,M);
x=x0;


%setup parameters for augmented Lagrangian
go=1;

k=0;
tveps=eps;
if verbose>4
    disp(eps)
end

while go
    k=k+1;

    % solve L2 subproblem  problem
    % min y    1/2 ||f(y)||_2^2 + tau/2 || y - b ||_2^2 
    %   with b = x+Multiplier/tau
    if k<6
        y=fun(tau,x-Multiplier/tau,tveps,x0);
    else
        y=fun(tau,x-Multiplier/tau,tveps,x);
    end
        
    % solve L1 subproblem  problem
    % min x    Phi(x) + tau/2 || x - rhs ||_2^2 
    %   with rhs = Multiplier/tau - y
    rhs=y+Multiplier/tau;
    
    switch reg
        case 0 % No denoising
            x=rhs;
        case 1 % Use derivative threshold
            x=Grad_denoise(rhs,lambda/tau); 
            % keyboard
        case 2 % TK denoising
            for i=1:M
                x(:,:,i)=(1+lambda/tau*(Lx)'*Lx+lambda/tau*(Ly)'*Ly)\rhs(:,:,i);
            end
        case 3 % TV denoising
            x=TV_denoise_group(rhs,lambda/tau,tveps);
    end
    
    go=0;
    if verbose>1
        fprintf('%5i ',k)
    end
    
    if verbose>1
        fprintf('%e ',max(abs(x(:)-y(:))))
    end
    
    if max(abs(x(:)-y(:)))>eps
        go=1;
    end
    
    Multiplier=Multiplier-1.8*tau*(x-y);
    if verbose>1
        fprintf('\n');
        if verbose>2
            figure(99)
            imagesc(abs(x(:,:,1))); colorbar
            drawnow;
        end
    end
    
    if k>Nmax 
        go=0;
    end
    
end
if verbose>0
    fprintf('%5i ',k)
    fprintf('%e ',max(abs(x(:)-y(:))))
    fprintf('\n');
end
