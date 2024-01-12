function pred = zeroNegEvalues( pred, num_hours )
%pred = zeroNegEvalues( pred, num_hours )

for fh = 1:num_hours
    inds = pred(fh).e < 0;
    pred(fh).e(inds) = 0;
end


end

