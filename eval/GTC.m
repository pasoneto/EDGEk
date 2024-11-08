cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator/present/new2/'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox2')

load mcdemodata

predicted_files_path = "/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/predicted_amass/";
pred_files = readtable("/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/random_selection_dances_perceptual_experiment.csv")
pred_files = pred_files.selected_dances;
true_files = dir("/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/positions_amass");
random_files = pred_files(randperm(length(pred_files)));

k=1
nameChosen = pred_files(k);
        
True = strcat(true_files(2).folder, "/", nameChosen);
pred = strcat(predicted_files_path, "", nameChosen);

True = readtable(True);
pred = readtable(pred);

True = table2array(True);
pred = table2array(pred);

%Adding to matlab object
df = dance1;
df.nFrames = height(True);
df.nMarkers = width(True)/3;
df.freq = 15;

trueD = df;
predE = df;

trueD.data = True;
predE.data = pred;
trueD.nMarkers=width(trueD.data)/3
predE.nMarkers=width(predE.data)/3

%Resampling to the same as AIST++
trueD.nFrames = height(trueD.data)
predE.nFrames = height(predE.data)

%Parameters for aist dataset
par = mcinitanimpar
par.msize = 8
par.output = "mp4";
par.videoformat = 'mp4'
par.conn = [1 2; 1 3; 1 4; 3 6; 2 5; 3 6; 4 7; 5 8; 6 9; 9 12; 8 11; 7 10; 10 13; 13 16; 10 14; 10 15; 14 17; 15 18; 18 20; 17 19; 20 22; 19 21; 21 23; 22 24];
par.markercolors='bbbbbbbbbbbbbbbbbbbbbbbb'        

par2 = mcinitanimpar
par2.msize = 8
par2.output = "mp4";
par2.videoformat = 'mp4'
par2.conn = [1 2; 1 3; 1 4; 3 6; 2 5; 3 6; 4 7; 5 8; 6 9; 9 12; 8 11; 7 10; 10 13; 13 16; 10 14; 10 15; 14 17; 15 18; 18 20; 17 19; 20 22; 19 21; 21 23; 22 24];
par2.markercolors='rrrrrrrrrrrrrrrrrrrrrrrr'

par3 = mcinitanimpar
par3.msize = 15
par3.output = "mp4";
par3.videoformat = 'mp4'
par3.conn = [];
par3.markercolors='r'
par3.trl=3
par3.trm=1
par3.tracecolors='r'

par4 = par3
par4.markercolors='b'
par4.tracecolors='b'

trueD = mccenter(trueD);
predE = mccenter(predE);

%CTC predicted
x = mean(predE.data(:, 1:3:72), 2);
y = mean(predE.data(:, 2:3:72), 2);
z = mean(predE.data(:, 3:3:72), 2);

gtcPred = predE;
gtcPred.data = [x y z];

gtcPred.nMarkers = 1;
gtcPred.markerName = trueD.markerName(1)

%GTC true
x = mean(trueD.data(:, 1:3:72), 2);
y = mean(trueD.data(:, 2:3:72), 2);
z = mean(trueD.data(:, 3:3:72), 2);

gtcTrue = trueD;
gtcTrue.data = [x y z];

gtcTrue.nMarkers = 1;
gtcTrue.markerName = trueD.markerName(1)

%Original animation
[all, allparams] = mcmerge(predE, mctranslate(trueD, [2 0 0]), par2, par);

%GTC animation
[allGTC, allparamsGTC] = mcmerge(gtcPred, mctranslate(gtcTrue, [2 0 0]), par3, par4);


%merge both
[all, allparams] = mcmerge(all, mctranslate(allGTC, [0 0 -1.5]), allparams, allparamsGTC);

all.freq = 15;
allparams.videoformat = "mp4";

nameChosen = "./gcs.mp4";

allparams.output = strrep(nameChosen, "csv", "mp4");    
mcanimate(all, allparams);

