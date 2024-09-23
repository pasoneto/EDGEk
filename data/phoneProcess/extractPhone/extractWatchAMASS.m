cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

%Convert Brigitta's data to csv format
directoryIn =  "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/sliced/";
directoryOut = "/Users/pdealcan/Documents/github/data/CoE/accel/amass/DanceDBPoses/watchPositions/";

files = dir(directoryIn);
for k=3:length(files)
    data = readtable(strcat(directoryIn, files(k).name));
    data = table2array(data);
    
    df = dance1;
    df.nMarkers = width(data)/3;
    df.nFrames = height(data);
    df.freq = 120;
    df.data = data;

%     df = mcresample(df, 30);

    %Saving phone IMUs           
    phone = (mcgetmarker(df, 3).data + mcgetmarker(df, 6).data)/2;
    watch = mcgetmarker(df, 21).data;

    IMU = [phone, watch];
    
    df.data = IMU;
    df.nMarkers=width(df.data)/3;
    df.nFrames = height(df.data);
    
%     par.markercolors = 'bbrbbrbbbbbbbbbbbbbbrbbb'
%     mcanimate(df)
    df = mcresample(df, 15);
    df = mctimeder(df, 2);
          
    IMU = df.data;
    IMU = array2table(IMU);     

    nameSave = strcat(directoryOut, files(k).name)

    writetable(IMU, nameSave);
end


