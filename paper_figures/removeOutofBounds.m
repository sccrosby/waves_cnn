function pred = removeOutofBounds( pred, ww3, num_hours, type )
%pred = removeOutofBounds( pred, ww3, num_hours, type )
% type = {'some','all'}

switch type
    case 'some'
        % Replace only moments thats are out of bounds
        for fh = 1:num_hours
            for ff = 1:length(pred(fh).fr)
                inds = abs(pred(fh).a1(:,ff)./pred(fh).e(:,ff)) > 1;
                pred(fh).a1(inds,ff) = ww3.a1(inds,ff);
                
                inds = abs(pred(fh).b1(:,ff)./pred(fh).e(:,ff)) > 1;
                pred(fh).b1(inds,ff) = ww3.b1(inds,ff);
                
                inds = abs(pred(fh).a2(:,ff)./pred(fh).e(:,ff)) > 1;
                pred(fh).a2(inds,ff) = ww3.a2(inds,ff);
                
                inds = abs(pred(fh).b2(:,ff)./pred(fh).e(:,ff)) > 1;
                pred(fh).b2(inds,ff) = ww3.b2(inds,ff);
                
            end
        end
    case 'all'        
        % Replace all moments if ANY moment is out of bounds
        for fh = 1:num_hours
            for ff = 1:length(pred(fh).fr)
                inds1 = abs(pred(fh).a1(:,ff)./pred(fh).e(:,ff)) > 1;
                
                inds2 = abs(pred(fh).b1(:,ff)./pred(fh).e(:,ff)) > 1;
                
                inds3 = abs(pred(fh).a2(:,ff)./pred(fh).e(:,ff)) > 1;
                
                inds4 = abs(pred(fh).b2(:,ff)./pred(fh).e(:,ff)) > 1;
                
                inds = inds1 | inds2 | inds3 | inds4;
                
                pred(fh).b2(inds,ff) = ww3.b2(inds,ff);
                pred(fh).a1(inds,ff) = ww3.a1(inds,ff);
                pred(fh).b1(inds,ff) = ww3.b1(inds,ff);
                pred(fh).a2(inds,ff) = ww3.a2(inds,ff);
                pred(fh).e(inds,ff) = ww3.e(inds,ff);
                
            end
        end
end


end

