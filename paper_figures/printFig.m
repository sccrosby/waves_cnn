function  printFig( hFig, fname, size, dtype, res )
%printFig( hFig, fname, size, dtype, res )
% e.g. printFig(gcf, 'MyFig', [10 6], 'png', 150)

dtype = validatestring(dtype,{'pdf','png'});

hFig.PaperUnits = 'inches';
hFig.PaperSize = size;
hFig.PaperPosition = [0 0 size];

switch dtype
    case 'pdf'
        print(hFig,'-dpdf',fname)
        
    case 'png'
        if nargin < 5
            res = 350; % Set default if not provided
        end
        res_str = sprintf('-r%d',res);
        print(hFig,'-dpng',res_str,fname)
end
        

end

