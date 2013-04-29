clear all
close all
addpath('..');
addpath('../../../common')


% n is the dimension of image and m is the number of angles for the projection;

m=32;
N=64;

% get a list of angles.
angist = linspace(-pi/2,pi/2, m+1); angist(end) = [];
% use the shear transform of a discrete image 
%to form a diagonal operator over the fourier transform domain. 
range = [0:N/2-1 0 -N/2+1:-1]';
[Y,X] = meshgrid(range,range);
shx = @(im,ang)real( ifft( fft(im) .* exp(-ang*2i*pi*X.*Y/N) ) );
shx = @(im,ang)fftshift(shx(fftshift(im),ang));
shy = @(im,ang)shx(im',ang)';
% now decompose rotation into operations of shear.
rots = @(f,t)shx( shy( shx(f,-tan(t/2)) ,sin(t)) ,-tan(t/2));
%define forward and inverse Radon transform
Phi  = @(fo) ex_radon_transform(fo, angist, +1, rots);
PhiS = @(ba) ex_radon_transform(ba, angist, -1, rots);

Phi = @(x) radon(x, linspace(0,180, m));

%% org and data
x0=phantom(N);
data=Phi(x0);
ds=size(data);


%% recon with aug. lagrangian
toVec=@(x)reshape(x,[N*N 1]);
toImg=@(x)reshape(x,[N N]);

fun2=@(tau,b,eps,x0)linls_cg(Phi,PhiS,data,tau,b,eps,50);
J=@(x)reshape(Phi(toImg(x))-data,[prod(ds) 1]);
fun5=@(tau,b,eps,x0)toImg(LBFGS(toVec(x0),@(y)fun_grad_FD(y,@(x)J(x),tau,toVec(b)),eps,20,20));

la=1e-4;
x=AugmentedLagrangian(fun5,[N N],struct('verbose',10,'tau',2*la,'lambda',la));
