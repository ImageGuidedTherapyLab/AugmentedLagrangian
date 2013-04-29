%% A class to handle 2D convolutions
% Example: A convolution of an image with a Gaussian can be written as:
%
% N=256;
% P = phantom(N);
% G = Convolution.Gauss([N N],.1)
% imagesc(G*P);
% 
% Convolution operators can also be composed and added
% Example:
% O= Convolution.OOF([N N],.1) % out of focus lense
% imagesc((G+O)*P); % or
% imagesc((G*O)*P);
% 
% Example Tikhonov regularized decomvolution:
%
% solve min_x || Ax-b ||^2_2 + lambda || \nabla x ||^2
% where A is a convolution operator
%
% The normal equation for this problem is
% (A'*A + lambda ( Dx'*Dx + Dy'*Dy ) ) * x = A'*b  
% where Dx and Dy are the derivative in x and y direction
%
% Code:
% P = phantom(N);
% A = Convolution.Gauss([N N],.1)
% b = A*P + 0.001* randn([N N]); % RHS is the blurred phantom with noise
%
% lambda =1e-2;
% Dx = Convolution.get_lx([N N]); % derivative operator in x-direction
% Dy = Convolution.get_ly([N N]);
% AA = A'*A + lambda *( Dx'*Dx + Dy'*Dy );
% Ab = A'*b;
% x_hat = AA\Ab;
% imagesc(x_hat);
%
% wstefan@mdanderson.org

classdef Convolution < handle
    properties
        PSF
        PSFh
    end
    
    methods
        function C=Convolution(PSF) % A convolution is defined by its PSF
            if exist('PSF','var')
                C.PSF=PSF;
                C.PSFh=fftn(PSF);
            end
        end
        
        function C=plus(A,B) % overwrite addition
            switch [class(A),class(B)]
                case ['Convolution','Convolution']
                    C=Convolution();
                    C.PSF=A.PSF+B.PSF;
                    C.PSFh=A.PSFh+B.PSFh;
                    % C=Convolution(A.PSF+B.PSF);
                case ['Convolution','double']
                    C=A+B*Convolution.eye(size(A.PSF)); 
                case ['double','Convolution']
                    C=B+A*Convolution.eye(size(B.PSF));
                otherwise
                    throw(MException('ResultChk:BadInput','Can not add classes %s and %s',class(A),class(B)))
            end
        end
        
        function C=minus(A,B)
            C = A+(-1)*B;
        end
        
        function C=mtimes(A,B) % overwride multiplication
            switch class(A)
                case 'double'
                    C=Convolution();
                    C.PSF=A*B.PSF;
                    C.PSFh=A*B.PSFh;  
                case 'Convolution'
                    switch class(B)
                        case 'Convolution'
                            C=Convolution();
                            C.PSFh=A.PSFh.*B.PSFh;
                            C.PSF=ifftn(C.PSFh);
                            % C=Convolution(ifft2(fft2(A.PSF).*fft2(B.PSF)));
                        case 'double'
                            if length(B)==1 % scalar multiplication
                                C=Convolution();
                                C.PSF=A.PSF*B;
                                C.PSFh=A.PSFh*B;
                                % C=Convolution(A.PSF*B);
                            else % Apply convolution operator to image
                                C=ifftn(A.PSFh.*fftn(B));
                            end
                        otherwise
                            throw(MException('ResultChk:BadInput','Can not multiply classes %s and %s',class(A),class(B)))
                            
                    end
                otherwise
                    throw(MException('ResultChk:BadInput','Can not multiply classes %s and %s',class(A),class(B)))
                    
            end
        end
        
        function C=ctranspose(A) % overwride transpose
            C=Convolution();
            C.PSFh = conj(A.PSFh);
            C.PSF = ifftn(C.PSFh);
            % C=Convolution(ifft2(conj(A.PSFh)));
        end
        
        function C=mldivide(A,B) % overwride \
            C=ifftn(fftn(B)./A.PSFh);
        end
        
    end
    
    methods(Static) % Static methods to generate paricular convolutions
        function C=Gauss(sz,sigma) % Gauss psf
            switch length(sz)
                case {1,2}
                    [X,Y]=meshgrid(linspace(-1,1,sz(2)),linspace(-1,1,sz(1)));
                    C=Convolution(fftshift(exp(-(X.^2+Y.^2)/sigma.^2)));
                case 3
                    [X,Y,Z]=meshgrid(linspace(-1,1,sz(2)),linspace(-1,1,sz(1)),linspace(-1,1,sz(1)));
                    C=Convolution(fftshift(exp(-(X.^2+Y.^2+Z.^2)/sigma.^2)));
            end       
        end
        
        function C=OOF(sz,r) % out of focus psf
            switch length(sz)
                case {1,2}
                    [X,Y]=meshgrid(linspace(-1,1,sz(2)),linspace(-1,1,sz(1)));
                    C=Convolution(double(fftshift(X.^2+Y.^2<r*r)));
                case 3
                    [X,Y,Z]=meshgrid(linspace(-1,1,sz(2)),linspace(-1,1,sz(1)),linspace(-1,1,sz(1)));
                    C=Convolution(double(fftshift(X.^2+Y.^2+Z.^2<r*r)));
            end
        end
        
        function C=eye(sz) % identity
            C=Convolution();
            C.PSF=zeros(sz);
            C.PSF(1)=1; 
            C.PSFh=ones(sz);
            
        end
        
        function C=get_l(sz,order,dir,type)
            % return the FD edge detectr with order (order) and in
            % y-direction (dir==0)
            % x-direction (dir==1)
            % z-direction (dir==2)
            % for an N(1)xN(2) image
            % type=0 is the psf
            % type=1 is the matching wave-form i.e. the unit jump response 

            if not(exist('type','var'))
                type=0;
            end
                
            c=zeros(order+1,1);
            for j=1:(order+1)
                d=j-(1:(order+1));d(d==0)=1;
                c(j)=factorial(order)/prod(d);
            end
            q=sum(c(1:(floor(order/2)+1)));
            c=c/q;
            switch dir
                case 0
                    n=sz(1);
                    psf0 = zeros(n,1);
                case 1
                    n=sz(2);
                    psf0 = zeros(n,1);
                case 2
                    if length(sz)<3
                        n=1;
                    else
                        n=sz(3);
                    end
                    psf0 = zeros(n,1);
            end

            if n==1
                psf0=0;
            else
                psf0(1:(order+1)) = c;
            end
            if type==0 
                lx = zeros(sz);
                switch dir
                    case 0
                        lx(:,1,1) = circshift(psf0,-floor(order/2));
                    case 1
                        lx(1,:,1) = circshift(psf0,-floor(order/2));
                    case 2
                        lx(1,1,:) = circshift(psf0,-floor(order/2));    
                end           
            else     
                % find the response to a 1d-unitjump of the PSF h
                % this is used to estimate the matching waveform of an edge detecor h
                psf1=psf0(end:-1:1);
                N=length(psf1);
                m=zeros(size(psf1));
                s=sum(psf1((N/2+1):N/2));
                for k=1:(N/2);
                    m(k)=s+sum(psf1(1:k));
                end
                for k=(N/2+1):N
                    m(k)=sum(psf1((N/2+1):k));
                end
                lx = zeros(sz);
                switch dir
                    case 0                  
                        lx(:,1,1) = circshift(m,order+1);
                    case 1                   
                        lx(1,:,1) = circshift(m,order+1);
                    case 2                   
                        lx(1,1,:) = circshift(m,order+1);    
                end
            end

            C=Convolution(lx);
        end

        function C=get_ly(sz)
            C=Convolution.get_l(sz,1,0);
        end
        
        function C=get_lx(sz)
            C=Convolution.get_l(sz,1,1);
        end      
        
        function C=get_lz(sz)
            C=Convolution.get_l(sz,1,2);
        end          
        
   end
end
            
            