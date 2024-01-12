function P = getndbc( id, yrstart )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%collate NDBC historical buoy data (only accesses data for the nearest whole year).
%DOES NOT use NetCDF instead downloads .txt summary files served by NDBC
%see http://www.ndbc.noaa.gov/measdes.shtml for measurement descriptions and units
%script also calculates wave power (wave energy flux) and adds it to the
%output
% Ian Miller, immiller@uw.edu
%VERSION UPDATE:  16 December 2016
% Modifications by sccrosby@ucsd.edu, 3/10/17


for station_index = 1;
    
    % Conversion Factors
    mbar2pa = 100; % mbar to Pa
    
    temp=datevec(today);
    yrend=temp(1); clear temp
    
    %initialize output structure
    StdMetData=struct('station',id,'time',[],'wvht',[],'dpd',[],'apd',[],...
        'mwvd',[],'wavepower',[],'winddir',[],'windspd',[],'gust',[],'pres',[],'airtemp',[],...
        'watertemp',[]);
    
    for i=yrstart:1:yrend
        url=['http://www.ndbc.noaa.gov/view_text_file.php?filename=' id 'h' num2str(i) '.txt.gz&dir=data/historical/stdmet/'];
        %url = sprintf('http://www.ndbc.noaa.gov/data/historical/stdmet/%sh%d.txt.gz',stname,i);
        sprintf('Reading %s data',num2str(i))
        try
            inr=webread(url);
            
            %read first line to get variable headers
            checkline=textscan(inr,'%s',1,'Delimiter','/n');
            
            %get number of variables in file and build appropriate format string for
            %textscan
            vars=strsplit(char(checkline{1}),' ');
            
            if length(vars)==14
                formstr='%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
            elseif length(vars)==15
                formstr='%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
            elseif length(vars)==16
                formstr='%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
            elseif length(vars)==17
                formstr='%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
            elseif length(vars)==18
                formstr='%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
            end
            
            %read in rest of data file as a series of strings
            data=textscan(inr,char(formstr),'headerlines',2,'EndofLine','\r\n');
            clear inr url %no longer needed so delete
            
            %% match date/time data in vars to output variable names for date/time
            
            if ~isempty(find(strncmp('YY',vars,2))) %this takes into account varying YR variable headers in NDBC text files
                yrs=data{find(strncmp('YY',vars,2))};
            else
                yrs=data{find(strncmp('#YY',vars,3))};
            end
            
            if nanmean(yrs)<100
                yrs=yrs+1900;
            elseif nanmean(yrs)<20
                yrs=yrs+2000;
            end
            
            mos=data{strmatch('MM',vars)};
            dy=data{strmatch('DD',vars)}; hrs=data{strmatch('hh',vars)};
            
            if strmatch('mm',vars) %this accounts for variation in ndbc text files - some include mintues, some don't
                dts=datenum(yrs,mos,dy,hrs,data{strmatch('mm',vars)},0);
            else
                dts=datenum(yrs,mos,dy,hrs,0,0);
            end
            
            StdMetData.time=[StdMetData.time; dts];
            
            %% now run through variable and assign to structure where possible
            
            %wave height
            if ~isempty(find(strncmp('WVHT',vars,4)))
                temp=data{find(strncmp('WVHT',vars,4))}; %wave height data
                temp(find(temp==99))=NaN;
                StdMetData.wvht=[StdMetData.wvht; temp];
                clear temp
            else
                StdMetData.wvht=[StdMetData.wvht; NaN(length(dts),1)];;
            end
            
            %dominant period
            if ~isempty(find(strncmp('DPD',vars,3)))
                temp=data{find(strncmp('DPD',vars,3))}; %wave height data
                temp(find(temp==99))=NaN;
                StdMetData.dpd=[StdMetData.dpd; temp];
                clear temp
            else
                StdMetData.dpd=[StdMetData.dpd; NaN(length(dts),1)];;
            end
            
            %average period
            if ~isempty(find(strncmp('APD',vars,3)))
                temp=data{find(strncmp('APD',vars,3))}; %wave height data
                temp(find(temp==99))=NaN;
                StdMetData.apd=[StdMetData.apd; temp];
                clear temp
            else
                StdMetData.apd=[StdMetData.apd; NaN(length(dts),1)];;
            end
            
            %mean wave direction
            if ~isempty(find(strncmp('MWD',vars,3)))
                temp=data{find(strncmp('MWD',vars,3))}; %wave height data
                temp(find(temp==999))=NaN;
                StdMetData.mwvd=[StdMetData.mwvd; temp];
                clear temp
            else
                StdMetData.mwvd=[StdMetData.mwvd; NaN(length(dts),1)];;
            end
            
            %wind direction
            if ~isempty(find(strncmp('WDIR',vars,4)))
                temp=data{find(strncmp('WDIR',vars,4))}; %wave height data
                temp(find(temp==999))=NaN;
                StdMetData.winddir=[StdMetData.winddir; temp];
                clear temp
            elseif ~isempty(find(strncmp('WD',vars,2)))
                temp=data{find(strncmp('WD',vars',2))};
                temp(find(temp==999))=NaN;
                StdMetData.winddir=[StdMetData.winddir; temp];
                clear temp
            else
                StdMetData.winddir=[StdMetData.winddir; NaN(length(dts),1)];
            end
            
            %wind speed
            if ~isempty(find(strncmp('WSPD',vars,4)))
                temp=data{find(strncmp('WSPD',vars,4))}; %wave height data
                temp(find(temp==99))=NaN;
                StdMetData.windspd=[StdMetData.windspd; temp];
                clear temp
            else
                StdMetData.windspd=[StdMetData.windspd; NaN(length(dts),1)];;
            end
            
            %wind gust
            if ~isempty(find(strncmp('GST',vars,3)))
                temp=data{find(strncmp('GST',vars,3))}; %wave height data
                temp(find(temp==99))=NaN;
                StdMetData.gust=[StdMetData.gust; temp];
                clear temp
            else
                StdMetData.gust=[StdMetData.gust; NaN(length(dts),1)];;
            end
            
            %atmospheric pressure
            if ~isempty(find(strncmp('PRES',vars,4)))
                temp=data{find(strncmp('PRES',vars,4))}; %wave height data
                temp(find(temp==9999))=NaN;
                StdMetData.pres=[StdMetData.pres; temp];
                clear temp
            elseif ~isempty(find(strncmp('BAR',vars,3)))
                temp=data{find(strncmp('BAR',vars',3))};
                temp(find(temp==9999))=NaN;
                StdMetData.pres=[StdMetData.pres; temp];
                clear temp
            else
                StdMetData.pres=[StdMetData.pres; NaN(length(dts),1)];
            end
            
            %air temperature
            if ~isempty(find(strncmp('ATMP',vars,4)))
                temp=data{find(strncmp('ATMP',vars,4))}; %wave height data
                temp(find(temp==999))=NaN;
                StdMetData.airtemp=[StdMetData.airtemp; temp];
                clear temp
            else
                StdMetData.airtemp=[StdMetData.airtemp; NaN(length(dts),1)];;
            end
            
            % water temperature
            if ~isempty(find(strncmp('WTMP',vars,4)))
                temp=data{find(strncmp('WTMP',vars,4))}; %wave height data
                temp(find(temp==999))=NaN;
                StdMetData.watertemp=[StdMetData.watertemp; temp];
                clear temp
            else
                StdMetData.watertemp=[StdMetData.watertemp; NaN(length(dts),1)];;
            end
        catch
            disp('no data for that year')
        end
    end
    
    
    %% store structure
    
    P.time = StdMetData.time;
    P.slp = StdMetData.pres*mbar2pa;
    P.wndspd = StdMetData.windspd;
    P.wnddir = StdMetData.winddir;
    P.airtemp = StdMetData.airtemp;
    P.seatemp = StdMetData.watertemp;
    P.waveheight = StdMetData.wvht;
    P.wavepeakperiod = StdMetData.dpd;
    P.wavemeanperiod = StdMetData.apd;
    P.wavedir = StdMetData.mwvd;
    P.wndgust = StdMetData.gust;
    
    
    
    
end

