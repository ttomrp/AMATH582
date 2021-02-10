%% GNR
close all; clear all; clc;
[y,Fs] = audioread('GNR.m4a');
L=length(y)/Fs; n=length(y);
S=y.';
t2=linspace(0,L,n+1); t=t2(1:n);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; ks=fftshift(k);
tau=100; tslide=0:0.1:L;
sgt_spec=zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-tau*(t-tslide(j)).^2);
    sg=g.*S;
    sgt=fft(sg);
    sgt_spec(j,:)=abs(fftshift(sgt));
end
pcolor(tslide, ks/(2*pi), sgt_spec'), shading interp
set(gca,'Ylim',[50 1000],'Fontsize',[12]) %guitar range
xlabel('Time(sec)'); ylabel('Frequency(Hz)'); colormap(hot)
%% Pink Floyd
close all; clear all; clc;
[y2,Fs] = audioread('Floyd.m4a');
y=y2(1:end-1); S=y.';
L=length(y)/Fs; n=length(y);
t2=linspace(0,L,n+1); t=t2(1:n);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; ks=fftshift(k);
tau=100; tslide=0:0.9:L;
sgt_spec=zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-tau*(t-tslide(j)).^2);
    sg=g.*S;
    sgt=fft(sg);
    sgt_spec(j,:)=abs(fftshift(sgt));
end
pcolor(tslide, ks/(2*pi), sgt_spec.'), shading interp
set(gca,'Ylim',[50 200],'Fontsize',[12]); %4string bass range
xlabel('Time(sec)'); ylabel('Frequency(Hz)'); colormap(hot);
%% Floyd, freq filter
close all; clear all; clc;
[y2,Fs]=audioread('Floyd.m4a');
y=y2(1:end-1); S=y.';
L=length(y)/Fs; n=length(y);
t2=linspace(0,L,n+1); t=t2(1:n);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; ks=fftshift(k);
tau=50; tslide=linspace(0,t(end),100);
sgt_spec=zeros(length(tslide),n);
for j=1:length(tslide)
    g=exp(-tau*(t-tslide(j)).^2);
    sgt=fft(g.*S);
    Sgts=abs(fftshift(sgt));
    [M,I]=max(Sgts(n/2:end));
    [I1,I2]=ind2sub(size(Sgts),I+n/2-1);
    g2=exp(-0.5*((ks-ks(I2)).^2));
    Sgts2=fftshift(sgt).*g2;
    sgt_spec(j,:)=abs(Sgts2);
end
pcolor(tslide, ks/(2*pi), log(sgt_spec.'+1)), shading interp
set(gca,'Ylim',[0 1000],'Fontsize',[12])
xlabel('Time(sec)'); ylabel('Frequency(Hz)'); colormap(hot)