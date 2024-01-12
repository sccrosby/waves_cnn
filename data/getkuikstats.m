function [md1,md2,spr1,spr2,skw,kur,sdmd1,sdspr1,spr2_H]=getkuikstats(a1,b1,a2,b2,dof)
% GETKUIKSTATS returns directional spectrum distribution stats in
% each freqeuncy band as defined by
% Kuik et. al., A method for routine analysis of pitch-and-roll
% data, JPO, 18, 1020-1034, 1988.
%
%
%  [md1,md2,spr1,spr2,skw,kur,sdmd1,sdspr1]=getkuikstats(a1,b1,a2,b2,dof)
% Or [md1,md2,spr1,spr2,skw,kur,sdmd1,sdspr1]=getkuikstats(d,dof)
% where d = [a1 b1 a2 b2] and is 4xN;
% input: low order normalized directional fourier coefficients a1,b1,a2,b2
%
% output:
%
%  md1= 1st moment mean direction 
%  md2= 2nd moment mean direction
%  spr1= 1st moment spread
%  spr2= 2nd moment spread
%  skw = skewness
%  kur = kurtosis
%  sdm1 = standard deviation of m1
%  sdspr1 = standard deviation of spr13

% Check number of inputs.
if nargin > 5  
    error('getkuikstats:TooManyInputs', ...
        'requires at most 5 inputs');
end

switch nargin
    case 4
        dof = 32;
    case 2
        dof = b1;
        b1 = a1(2,:); a2 = a1(3,:); b2 = a1(4,:); a1 = a1(1,:);
    case 1
        b1 = a1(2,:); a2 = a1(3,:); b2 = a1(4,:); a1 = a1(1,:); 
        dof = 32;
end



% first moment mean direction
% radians
md1r=atan2(b1+eps,a1+eps);

% degrees
md1=md1r*(180/pi);
% turn negative directions in positive directions
md1(md1 < 0)=md1(md1 < 0)+360;

% first moment spread
spr=2*(1-sqrt(a1.^2+b1.^2));
spr1=sqrt(spr)*180/pi;

% second moment mean direction in degrees
md2=0.5*atan2(b2,a2)*(180/pi);
% turn negative directions in positive directions
md2(md2 < 0)=md2(md2 < 0)+360;
% a2b2 mean dir has 180 deg amiguity. find one that is closest to
% a1b1 mean dir.
tdif=abs(md1-md2);
md2(tdif > 90)=md2(tdif > 90)-180;
md2(md2 < 0)=md2(md2 < 0)+360;

% second moment spread  
m2=a2.*cos(2*md1r)+b2.*sin(2*md1r);
spr2=sqrt((1.0-m2)/2)*(180/pi);

spr2_H = 180/pi*sqrt((1-sqrt(a2.^2+b2.^2))/2);

% skewness & kurtosis

% m1 after Kuik et. al.
rm1=sqrt(a1.^2+b1.^2);
% 2 times mean direction
t2=2*atan2(b1,a1);
% n2 after Kuik et. al.
rn2=b2.*cos(t2)-a2.*sin(t2);
% m2 after Kuik et. al.
rm2=a2.*cos(t2)+b2.*sin(t2);

% kurtosis_1
kur=(6.-8.*rm1+2.*rm2)./((2*(1.-rm1)).^2);
% skewness_1
skw=-rn2./(.5*(1-rm2)).^1.5;


% Use Kuik, 1988 Eqs 40,41 to estimate standard deviation which is very
% near to rms error as bias is an order of magnitude less
% s.d.(md1)
sdmd1 = dof^(-.5)*sqrt((1-rm2)./(2*rm1.^2));
sdmd1 = sdmd1*(180/pi);
% s.d.(spr1)
sdspr1 = dof^(-.5)*sqrt(rm1.^2./(2*(1-rm1)).*(rm1.^2+...
    (rm2.^2+rn2.^2-1)/4+(1+rm2).*(rm1.^(-2)-2)/2));
sdspr1 = sdspr1*(180/pi);

end