function [E, e] = reconstruct_2d_spectra(hs,tp,dspr,md,jon_gam,f,theta,bw,dtheta)
%[E, e] = reconstruct_2d_spectra(hs,tp,dspr,md,jon_gam,f,theta,bw,dtheta)
% S. C. Crosby
% See Kumar, N., Cahl, D. L., Crosby, S. C., &#38; Voulgaris, G. (2017). Bulk versus Spectral Wave Parameters: Implications on Stokes Drift Estimates, Regional Wave Modeling, and HF Radars Applications. <i>Journal of Physical Oceanography</i>

% Example settings
% hs = 2;
% tp = 10;
% dspr = 20*2*pi/180;
% md = 270;
% jon_gam = 3.3;
% f = .035:.005:1;
% bw = diff(f(1:2));
% theta = 0.5:359.5;
% dtheta = pi/180;
% E = reconstruct_2d_spectra(hs,tp,dspr,md,jon_gam,f,theta,bw,dtheta)

% jonswap e(f)
e = jonswap2(f,'Hm0',hs,'Tp',tp,'gamma',jon_gam);

% cos2n D(theta)
s = 2/dspr^2-1;
%dcoef = 2*2^(2*s-1)/pi*gamma(s+1)^2/gamma(2*s+1); %s gets too big for gamma()
D = cosd(theta-md).^(2*s);
pleft = wrapTo360(round(md)-90);
pright = wrapTo360(round(md)+90);
if md < 90 || md > 270   
    inds = pright:pleft;
else
    inds = [1:pleft pright:360];
end
D(inds) = 0;
D = D/(sum(D)*dtheta);

% Check
if abs(sum(D*dtheta)-1) > .0001
    error('ERROR, direc spect is not correctly normalized')
end

% E(f,theta)
E = e'*D;

% Check
hs_check = 4*sqrt(sum(E(:))*dtheta*bw);

if abs(hs-hs_check) > 0.05
    error('ERROR, integrated hs not the same as input')
end

end

