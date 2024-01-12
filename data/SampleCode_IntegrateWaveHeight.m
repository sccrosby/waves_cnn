%% Sample code
%% clearvars
clear

O = load('CDIPObserved_46211_hourly.mat');
P = load('WW3CFSRphase2_46211_rebanded.mat');

%%

% integration across direction to E(time,fr)
P.e = squeeze(sum(P.sp,2)*P.dtheta);
% integration across frequency to Hs
P.hs = 4*sqrt(P.e'*P.bw);

% integration across frequency to Hs
O.hs = 4*sqrt(O.e'*O.bw);

% Transform from WW3 Spectra to buoy observed moments (Fourier moments)
for tt = 1:length(P.time)
P.a1(:,tt) = P.sp(:,:,tt)*cosd(P.dir)*P.dtheta;
P.b1(:,tt) = P.sp(:,:,tt)*sind(P.dir)*P.dtheta;
P.a2(:,tt) = P.sp(:,:,tt)*cosd(2*P.dir)*P.dtheta;
P.b2(:,tt) = P.sp(:,:,tt)*sind(2*P.dir)*P.dtheta;
end

% Save the moments for WW3
% [('time', 'O'), ('sp', 'O'), ('dir', 'O'), ('bw', 'O'), ('fr', 'O'), ('lat', 'O'), ('lon', 'O'), ('dtheta', 'O'), ('e', 'O'), ('hs', 'O'), ('a1', 'O'), ('b1', 'O'), ('a2', 'O'), ('b2', 'O')]
%save('WW3CFSRphase2_46211_rebanded_moments.mat', '-struct', 'P')
%break;
%return;

% Estimate bulk direction statistics
[O.md1,O.md2,O.spr1,O.spr2,O.skw,O.kur]=getkuikstats(O.a1./O.e,O.b1./O.e,O.a2./O.e,O.b2./O.e);
[P.md1,P.md2,P.spr1,P.spr2,P.skw,P.kur]=getkuikstats(P.a1./P.e,P.b1./P.e,P.a2./P.e,P.b2./P.e);


%% Wave height comparison
clf
subplot(511)
plot(O.time,O.hs)
hold on
plot(P.time,P.hs)
legend('Obs','WW3')
ylabel('Hs [m]')
datetick('x')

%% Energy and moments
% Pick frequency to plot
ff = 15;

clf
subplot(511)
plot(O.time,O.e(ff,:))
hold on
plot(P.time,P.e(ff,:))
legend('Obs','WW3')
ylabel('Hs [m]')
datetick('x')
title(sprintf('Freq = %4.3f Hz',O.fr(ff)))

subplot(512)
plot(O.time,O.a1(ff,:))
hold on
plot(P.time,P.a1(ff,:))
legend('Obs','WW3')
ylabel('a1')
datetick('x')

subplot(513)
plot(O.time,O.b1(ff,:))
hold on
plot(P.time,P.b1(ff,:))
legend('Obs','WW3')
ylabel('b1')
datetick('x')

subplot(514)
plot(O.time,O.a2(ff,:))
hold on
plot(P.time,P.a2(ff,:))
legend('Obs','WW3')
ylabel('a2')
datetick('x')

subplot(515)
plot(O.time,O.b2(ff,:))
hold on
plot(P.time,P.b2(ff,:))
legend('Obs','WW3')
ylabel('b2')
datetick('x')


%% Compare mean direction and spread (derivative of fourier moments)
subplot(411)
plot(O.time,O.md1(ff,:))
hold on
plot(P.time,P.md1(ff,:))
legend('Obs','WW3')
ylabel('dir1')
datetick('x')
ylim([200 300])

subplot(412)
plot(O.time,O.md2(ff,:))
hold on
plot(P.time,P.md2(ff,:))
legend('Obs','WW3')
ylabel('dir2')
datetick('x')
ylim([200 300])

subplot(413)
plot(O.time,O.spr1(ff,:))
hold on
plot(P.time,P.spr1(ff,:))
legend('Obs','WW3')
ylabel('spr1')
datetick('x')
%ylim([200 300])

subplot(414)
plot(O.time,O.spr2(ff,:))
hold on
plot(P.time,P.spr2(ff,:))
legend('Obs','WW3')
ylabel('spr2')
datetick('x')