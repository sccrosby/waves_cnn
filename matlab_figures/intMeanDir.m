function data = intMeanDir( data, num_hours, bnds )
%data = intMeanDir( data, num_hours, bnds )


% Average (multiple lines important!)
for fh = 1:num_hours

% a1N = data(fh).a1(:,bnds)./data(fh).e(:,bnds);
% clf
% plot(a1N)
% ylim([-5 5])
% return
ei = sum(data(fh).e(:,bnds),2);
a1i = sum(data(fh).a1(:,bnds),2);
a1iN = a1i./ei;
b1i = sum(data(fh).b1(:,bnds),2);
b1iN = b1i./ei;
a2i = sum(data(fh).a2(:,bnds),2);
a2iN = a2i./ei;
b2i = sum(data(fh).b2(:,bnds),2);
b2iN = b2i./ei;

[data(fh).md1,data(fh).md2,data(fh).spr1,data(fh).spr2,~,~]=getkuikstats(a1iN,b1iN,a2iN,b2iN);
end

end

