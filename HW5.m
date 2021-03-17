close all; clear; clc;
Video=VideoReader('ski_drop_low.mp4');
%Video=VideoReader('monte_carlo_low.mp4');
for i=1:1:Video.NumFrames
    grayFrame=rgb2gray(read(Video,i));
    Frame=reshape(grayFrame,Video.Width*Video.Height,1);
    Frames(:,i)=double(Frame);
end
t2=linspace(0,Video.CurrentTime, Video.NumFrames+1); t=t2(1:end-1);
dt=t(2)-t(1);
%% DMD
X1=Frames(:,1:end-1); X2=Frames(:,2:end);
[U,S,V]=svd(X1,'econ');
lambda=diag(S).^2; 
figure(1) %singular values
subplot(2,1,1), plot(diag(S)/sum(diag(S)),'ko-','Linewidth',[1])
title('Singular Values')
subplot(2,1,2), plot(lambda/sum(lambda)*100,'ko-','Linewidth',[1])
title('Percentage Variance')

r=1; % rank 1
U=U(:,1:r); S=S(1:r,1:r); V=V(:,1:r);
A=U'*X2*V*diag(1./diag(S)); % low rank aproximation
[eigVec,eigVal]=eig(A);
Phi=X2*V/S*eigVec; % DMD modes
% low-rank DMD
mu=diag(eigVal);
omega=log(mu)/dt;
x1=X1(:,1); y0=Phi\x1;
modes=zeros(r,length(t));
modes(:,1)=y0;
for j=1:length(t)-1
    modes(:,j)=(y0.*exp(omega*t(j)));
end
Xdmd=Phi*modes;
% sparse DMD
Xsparse=(Frames-abs(Xdmd)); %add positive values for more contrast
R=Xsparse.*(Xsparse<0.01);
Xdmd=R+abs(Xdmd);
Xsparse=Xsparse-R;
%%
k=100;
figure(2)
subplot(3,1,1)
imshow(reshape(uint8(Frames(:,k)),Video.Height,Video.Width))
title('Original')
subplot(3,1,2)
imshow(reshape(uint8(Xdmd(:,k)),Video.Height,Video.Width))
title('Low-rank DMD')
subplot(3,1,3)
imshow(reshape(uint8(Xsparse(:,k)),Video.Height,Video.Width))
title('Sparse DMD')
