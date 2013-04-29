function y = ex_radon_transform(x, tlist, direction, rotation)

n = size(x,1);
m = length(tlist);

if nargin<3
    t = linspace(-1,1,n);
    [Y,X] = meshgrid(t,t);
    rotation = @(f,t)interp2(Y,X,f, sin(t)*X + cos(t)*Y, cos(t)*X - sin(t)*Y, 'cubic', 0); 
end

if direction==1
    %% Radon %%
    y = zeros(n,m);
    for i=1:m
        y(:,i) = sum(rotation(x, tlist(i)), 2);
    end
else
    %% Adjoint %%
    y = zeros(n);
    for i=1:m
        y = y + rotation(repmat(x(:,i), [1 n]), -tlist(i));
    end
end
