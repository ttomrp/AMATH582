close all; clear all; clc
load subdata.mat
L=10; % spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); xp=x2(1:n);
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);
[X,Y,Z]=meshgrid(xp,xp,xp);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);
%graph of unfiltered unaveraged data
Un(:,:,:)=reshape(subdata(:,1),n,n,n);
close all, isosurface(X,Y,Z,abs(Un),0.4)
axis([-10 10 -10 10 -10 10]), grid on, drawnow
%title('Noisy Data')
xlabel('X'), ylabel('Y'), zlabel('Z')
%% average the signal
Uavg=zeros(n,n,n);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Uavg = Uavg + fftn(Un);
end
Uavg=abs(fftshift(Uavg))./49;
[M,I]=max(Uavg(:)); 
%% graph of averaged unfiltered spectral data
figure(2)
isosurface(Kx,Ky,Kz,abs(Uavg)./abs(M), 0.2)
axis([-10 10 -10 10 -10 10]), grid on, drawnow
%title('Isosurface of Averaged Spectral Data')
xlabel('Wavenumber (k)'),ylabel('Wavenumber (k)'),zlabel('Wavenumber (k)') 
%% central freq and filter
[I1,I2,I3]=ind2sub(size(Uavg),I);
fx=Kx(I1,I2,I3); fy=Ky(I1,I2,I3); fz=Kz(I1,I2,I3);
filter=exp(-((Kx-fx).^2 + (Ky-fy).^2 + (Kz-fz).^2)); % gauss filter
%% positions of the sub
xpos = zeros(1,49); ypos = zeros(1,49); zpos = zeros(1,49);
for j=1:49
    Un2(:,:,:)=fftn(reshape(subdata(:,j),n,n,n));
    Unft=filter.*fftshift(Un2);
    Unfs=ifftshift(Unft);
    Unf=ifftn(Unfs);
    Unf=ifftn(Unft);
    [M,I]=max(abs(Unf(:)));
    [I1,I2,I3]=ind2sub(size(Unf),I);
    xpos(j)=X(I1,I2,I3);ypos(j)=Y(I1,I2,I3);zpos(j)=Z(I1,I2,I3);
end
end_position = [xpos(49), ypos(49), zpos(49)] % ending sub position
%% plot the path of the sub
plot3(xpos, ypos, zpos,'-o','Linewidth',[1]), grid on;
hold on
plot3(xpos(49),ypos(49),zpos(49),'r*');
%title('Sub Path')
xlabel('X'), ylabel('Y'), zlabel('Z')