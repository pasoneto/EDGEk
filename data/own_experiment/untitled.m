cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox')
addpath('/Users/pdealcan/Documents/github/matlabTools/MIRtoolbox/MIRToolbox')

load mcdemodata

dirIn = "/Users/pdealcan/Documents/github/EDGEk/data/own_experiment/";

phone = readtable(strcat(dirIn, "Accelerometer.csv"));
phone = removevars(phone, 'time');

wrist = readtable(strcat(dirIn, "WristMotion.csv"));
wrist = wrist(:, {'seconds_elapsed', 'accelerationX', 'accelerationY', 'accelerationZ'});

phone = table2array(phone);
wrist = table2array(wrist);
    
df = dance1;
df.nFrames = height(phone);
df.freq = round(length(phone(:,1))/max(phone(:,1)));
phone = phone(:,2:end);
df.data = phone;
df.nMarkers = width(phone)/3;

df2 = dance1;
df2.nFrames = height(wrist);
df2.freq = round(length(wrist(:,1))/max(wrist(:,1)));
wrist = wrist(:,2:end);
df2.data = wrist;
df2.nMarkers = width(wrist)/3;

df = mcresample(df, 15);
df2 = mcresample(df2, 15);

df = mctrim(df, 2, 12);
df2 = mctrim(df2, 2, 12);

IMU = [df.data df2.data];

writetable(IMU, nameSave);

