clear all
close all
addpath('..');
addpath('../../../common')


% n is the dimension of image and m is the number of angles for the projection;

m=32;
n=128;

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


%% org and data
x0=phantom(n);
data=Phi(x0);


%% recon with aug. lagrangian

fun2=@(tau,b,eps)linls_cg(Phi,PhiS,data,tau,b,eps,50);

la=1e-4;
x=TV_min(fun2,[n n],struct('verbose',10,'tau',2*la,'lambda',la));
