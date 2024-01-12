function [ wndspdi, wnddiri ] = interpwinds( time, wndspd, wnddir, newtime, convlen )
%[ wndspdi, wnddiri ] = interpwinds( time, wndspd, wnddir, newtime )
%  Interp in u,v space
%   S. C. Crosby

u = wndspd.*cosd(wnddir);
v = wndspd.*sind(wnddir);

u = conv(u,1/convlen*ones(convlen,1),'same');
v = conv(v,1/convlen*ones(convlen,1),'same');

[~,I] = unique(time);

ui = interp1(time(I),u(I),newtime);
vi = interp1(time(I),v(I),newtime);

wndspdi = hypot(ui,vi);
wnddiri = atan2d(vi,ui);
wnddiri = wrapTo360(wnddiri);

end

