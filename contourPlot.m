function []=contourPlot()
%plot contours of joint pdfs of two betas of the following distributions:
%Gaussian,Laplace,SymGamma, Alpha-stable

close all;
clc;

contourLevel=1/(2*pi*exp(1));
cBegin=.003;
cEnd=0.05;
nContour=5;

contourLevelV=exp(linspace(log(cBegin),log(cEnd),nContour));
figure(1)

for i=1:nContour
    %Gauss
    contourLevel=contourLevelV(i);
    subplot(2,2,1)
    box on;
    title('Gaussian')
    plotGauss(contourLevel)
    axis equal
    ylim([-4,4])
    xlim([-4,4])
end

for i=1:nContour
    %Laplace

    hold on
    contourLevel=contourLevelV(i);
    subplot(2,2,2)
    title('Laplace')
    plotLaplace(contourLevel)
    axis equal
    box on
    ylim([-9,9])
    xlim([-9,9])
end

for i=1:nContour
    %SymGamma
    hold on
    cEnd = 10;
    contourLevelV=exp(linspace(log(cBegin),log(cEnd),nContour));
    contourLevel=contourLevelV(i);
    subplot(2,2,3)
    title('Symmetric Gamma')
    plotSymGamma(contourLevel)
    axis equal
    box on
    ylim([-2,2])
    xlim([-2,2])
end

for i=1:nContour
    %Stable
    hold on
    contourLevel=contourLevelV(i);
    subplot(2,2,4)
    title('Symmetric \alpha-Stable')
    plotStable(contourLevel)
    axis equal
    box on
    nWindow=2;
    ylim([-nWindow,nWindow])
    xlim([-nWindow,nWindow])
end


end

function[]=plotGauss(contourLevel)
nPlot=1000;
lambda2=0.5;
r=sqrt((2/lambda2)*log(1/(2*pi*contourLevel)));
h=circle(0,0,r);
end

function[]=plotLaplace(contourLevel)
lambda=1;
intVal=(2/lambda)*log(1/(4*contourLevel));

hold on
plot([intVal 0],[0, intVal],'b')
plot([-intVal 0],[0, intVal],'b')
plot([intVal 0],[0, -intVal],'b')
plot([-intVal 0],[0, -intVal],'b')

end

function h = circle(x,y,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit,'g');
end

function []=plotSymGamma(contourLevel)
beta1Start=0.001;
beta1End=3.5;

nPts=1000;

beta1Vec=linspace(beta1Start,beta1End,nPts);
beta2Vec=zeros(1,nPts);
eta=5;
for i=1:nPts
    b1=beta1Vec(i);
    func=@(b2) (1/(b2*beta1Vec(i)))*exp(-eta*(b2+beta1Vec(i)))-contourLevel;
    if i == 1
        initVal=-log(contourLevel*b1)/eta;
    else
        initVal=beta2Vec(i-1);
    end
%     if b1<1
%         initVal=-log(contourLevel*b1)/eta;
%     else
%         initVal=1/(b1*contourLevel)*exp(-eta*b1);
%     end
    [beta2Vec(1,i)] = fzero(func,initVal);
end
hold on
plot(beta1Vec,beta2Vec,'r')
plot(beta1Vec,-beta2Vec,'r')
plot(-beta1Vec,beta2Vec,'r')
plot(-beta1Vec,-beta2Vec,'r')
end

function []=plotStable(contourLevel)
beta1Start=0.03;
beta1End=3.5;

nPts=1000;

beta1Vec=linspace(beta1Start,beta1End,nPts);
beta2Vec=zeros(1,nPts);
eta=5;

cAlpha=(1/pi^2);
for i=1:nPts
    b1=beta1Vec(i);
    [beta2Vec(1,i)] =(1/(pi*b1))*sqrt(contourLevel);
end
hold on
plot(beta1Vec,beta2Vec,'k')
plot(beta1Vec,-beta2Vec,'k')
plot(-beta1Vec,beta2Vec,'k')
plot(-beta1Vec,-beta2Vec,'k')
end