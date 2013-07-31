function x=TV_denoise_group(y,lambda,eps)
    %% TV denoising
    % solves min_x 1/2 || x- y || + lambda TV(x)
    % y input images (Nx x Ny x M)
    % TV(x) = || sqrt( (Lx * x(:,:,1) , Ly * x(:,:,1), Lx * x(:,:,2) , Ly *
    % x(:,:,2), ... ).^2 ) ||_1
    % lambda parameter. Larger means more denoising.
    % stopping eps 

    if not(exist('eps','var'))
        eps=1e-3;
    end
    
    [Ny Nx M]=size(y);
    Lx=Convolution.get_lx([Ny Nx]);
    Ly=Convolution.get_ly([Ny Nx]);

    w=zeros([Ny Nx M 2]);
    bs=zeros([Ny Nx M 2]);
    LLx=zeros([Ny Nx M 2]);
    Lambda=zeros([Ny Nx M 2]);
    tau=max(lambda(:))*20;
    lhs=1+tau*Lx'*Lx+tau*Ly'*Ly;
    go=1; i=0;
    epsno=norm(y(:));
    while go
        i=i+1;
        x=zeros(Ny,Nx,M);
        for j=1:M
            rhs=y(:,:,j)+Lx'*(Lambda(:,:,j,1)+tau*w(:,:,j,1))+Ly'*(Lambda(:,:,j,2)+tau*w(:,:,j,2));
            x(:,:,j)=lhs\rhs;
        end
        for j=1:M
            LLx(:,:,j,1)=Lx*x(:,:,j);
            LLx(:,:,j,2)=Ly*x(:,:,j);
        end
        b=LLx-Lambda/tau;
        
        % Soft threshold
        ba=sqrt(sum(sum(conj(b).*b,3),4));
        
        for j=1:M
            bs(:,:,j,1)=b(:,:,j,1)./ba;
            bs(:,:,j,2)=b(:,:,j,2)./ba;
        end
        ba=max(ba-lambda/tau,0);
        for j=1:M
            w(:,:,j,1)=bs(:,:,j,1).*ba;
            w(:,:,j,2)=bs(:,:,j,2).*ba;
        end

        d=LLx-w;
        Lambda=Lambda-1.8*tau*d;
        go=and((norm(d(:))/epsno)>eps,i<500);
        if mod(i,10)==0;
            tau=tau*2;
            lhs=1+tau*Lx'*Lx+tau*Ly'*Ly;
        end
        
        fprintf('%i: %e\n',i,norm(d(:))/epsno)
        %figure(99); imagesc(abs(x));
        %drawnow
    end
        
        