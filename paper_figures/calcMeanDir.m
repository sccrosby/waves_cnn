function data = calcMeanDir( data, num_hours, Nf )

% 
for fh = 1:num_hours
    for ff = 1:Nf
        ei = sum(data(fh).e(:,ff),2);
        a1i = sum(data(fh).a1(:,ff),2);
        a1iN = a1i./ei;
        b1i = sum(data(fh).b1(:,ff),2);
        b1iN = b1i./ei;
        a2i = sum(data(fh).a2(:,ff),2);
        a2iN = a2i./ei;
        b2i = sum(data(fh).b2(:,ff),2);
        b2iN = b2i./ei;        
        [data(fh).md1(:,ff),data(fh).md2(:,ff),data(fh).spr1(:,ff),data(fh).spr2(:,ff),~,~]=getkuikstats(a1iN,b1iN,a2iN,b2iN);
    end
end

end

