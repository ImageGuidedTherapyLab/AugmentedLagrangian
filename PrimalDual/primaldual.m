clear all;
n = 128;
name = 'phantom';
f0 = load_image(name, n);

% n is the dimension of image and m is the number of angles for the projection;
% get a list of angles.
angist = linspace(-pi/2,pi/2, m+1); angist(end) = [];
% use the shear transform of a discrete image 
%to form a diagonal operator over the fourier transform domain. 
range = [0:n/2-1 0 -n/2+1:-1]';
[Y,X] = meshgrid(range,range);
shx = @(im,ang)real( ifft( fft(im) .* exp(-ang*2i*pi*X.*Y/n) ) );
shx = @(im,ang)fftshift(shx(fftshift(im),ang));
shy = @(im,ang)shx(im',ang)';
% now decompose rotation into operations of shear.
rots = @(f,t)shx( shy( shx(f,-tan(t/2)) ,sin(t)) ,-tan(t/2));
%define forward and inverse Radon transform
Phi  = @(fo) ex_radon_transform(fo, angist, +1, rots);
PhiS = @(ba) ex_radon_transform(ba, angist, -1, rots);



% consider noiseless measurements y=Φf0.
y = Phi(f0);
tb = @(x) reshape(Phi(PhiS(reshape(x,n,m))), n*m, 1);
niter = 10;
[y1,FLAG,RELRES,ITER,RESVEC] = cgs(tb,y(:),1e-10,niter);
y1 = reshape(y1,n,m);
fre = PhiS(y1);
figure(1);
imagesc(fre);


%The primal-dual algorithm 
Amplitude = @(u) sqrt(sum(u.^2,3));
F1 = @(u) sum(sum(Amplitude(u)));
u = @(z) reshape(z(1:n*m),n,m);
v = @(z) reshape(z(n*m+1:end),n,n,2);
Ko1 = @ (fa) reshape(Phi(fa),n*m,1);
Kt2 = @ (fa) reshape(grad(fa), n*n*2,1);
Kt3 = @ (fa) reshape(grad(fa), n,n,2);
uu = @ (yj) yj(1:n*m,1);
uuu = @ (yy) reshape(yy,n*m,1);
vv = @ (yj) yj(n*m+1:end,1);
vvv = @ (yy) reshape(yy,n*n,1);
zzz= @ (yj,fa,lambda,sigma) (vv(yj)+sigma*Kt2(fa))./repmat(max([zeros(n*n,1)+lambda vvv(Amplitude(reshape(vv(yj),n,n,2)+Kt3(fa)))], [],2),  [2 1]);
xxx= @ (yj,fa,lambda,sigma) (uu(yj)+sigma*(Ko1(fa)-uuu(y)))/(1+sigma);
ProxFS = @(yj,fa,lambda,sigma) [xxx(yj,fa,lambda,sigma); zzz(yj,fa,lambda,sigma)];
ProxG = @(x,lambdaa) x;
K  = @(f) [reshape(Phi(f),n*m,1); reshape(grad(f), n*n*2,1)];
KS = @(z) PhiS(u(z)) - div(v(z));
% parameters. L=||K||2=nm. One should has Lσλτ<1.
L = n*m;
sigma = 10;
lambda=1;
% sigma = 1;
tau = .9/(L*sigma*lambda);
theta = 1;

f = fL2;
f=f*0;
g = K(f)*0;
f1 = f;
niter = 10;
% niter = 200;
% G = []; C = []; F = [];
for i=1:niter
    % update
    fold = f;
    g = ProxFS( g,f1, lambda, sigma);
    f = ProxG(  f-tau*KS(g), tau);
    f1 = f + theta * (f-fold);
    % monitor the decay of the energy
end

figure(2);
imagesc((f1));
figure(3);
imagesc((f));
