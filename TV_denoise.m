function x=TV_denoise(y,lambda,eps)
    %% TV denoising
    % solves min_x 1/2 || x- y || + lambda TV(x)
    % y input image
    % lambda parameter. Larger means more denoising.
    % stopping eps 

    if not(exist('eps','var'))
        eps=1e-3;
    end
    
    [Ny Nx]=size(y);
    Lx=Convolution.get_lx([Ny Nx]);
    Ly=Convolution.get_ly([Ny Nx]);

    w=zeros([Ny Nx 2]);
    bs=zeros([Ny Nx 2]);
    LLx=zeros([Ny Nx 2]);
    Lambda=zeros([Ny Nx 2]);
    tau=lambda*20;
    lhs=1+tau*Lx'*Lx+tau*Ly'*Ly;
    go=1; i=0;
    epsno=norm(y(:));
    while go
        i=i+1;
        rhs=y+Lx'*(Lambda(:,:,1)+tau*w(:,:,1))+Ly'*(Lambda(:,:,2)+tau*w(:,:,2));
        x=lhs\rhs;
        
        LLx(:,:,1)=Lx*x;
        LLx(:,:,2)=Ly*x;
        b=LLx-Lambda/tau;
        
        % Soft threshold
        ba=sqrt(sum(conj(b).*b,3));
        bs(:,:,1)=b(:,:,1)./ba;
        bs(:,:,2)=b(:,:,2)./ba;
        ba=max(ba-lambda/tau,0);
        w(:,:,1)=bs(:,:,1).*ba;
        w(:,:,2)=bs(:,:,2).*ba;
        
        d=LLx-w;
        Lambda=Lambda-1.8*tau*d;
        go=and((norm(d(:))/epsno)>eps,i<500);
        if mod(i,10)==0;
            tau=tau*2;
            lhs=1+tau*Lx'*Lx+tau*Ly'*Ly;
        end
        
        %fprintf('%i: %e\n',i,norm(d(:))/epsno)
        %figure(99); imagesc(abs(x));
        %drawnow
    end
        
        