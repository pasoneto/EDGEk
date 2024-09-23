cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator/present/new2/'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox2')

load mcdemodata

predicted_files = dir("/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/predicted_amass/");
true_files = dir("/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/positions_amass/");

strList = {true_files(~[true_files.isdir]).name};
% Initialize an empty cell array to store the results
result = cell(size(strList));

substrings = {'sliced_0_', 'Angry', 'Curiosity', 'Happy', 'Nervous', 'Sad', 'Scary', 'Excited', 'Annoyed', 'Bored', 'Miserable', 'Mix', 'Pleased', 'Relaxed', 'Sad', 'Satisfied', 'Tired', 'Afraid', 'Neutral'};

% Loop through each string in the list
for i = 1:length(strList)
    % Split the string by underscores
    parts = split(strList{i}, '_');  
    if length(parts) > 2
        b = strjoin(parts(3:end), '_');
        result{i} = b;
    else
        result{i} = '';
        disp("Found error here")
        disp(b)
    end
end

% Display the result
all_videos = unique(result);

% for k=3:length(predicted_files)
k = 108

nameChosen = strcat("sliced_5_", all_videos(k));

True = strcat(true_files(2).folder, "/", nameChosen);
pred = strcat(predicted_files(2).folder, "/", nameChosen);

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
trueD.nMarkers=width(trueD.data)/3;
predE.nMarkers=width(predE.data)/3;

%Resampling to the same as AIST++
trueD.nFrames = height(trueD.data);
predE.nFrames = height(predE.data);

%Parameters for aist dataset
par = mcinitanimpar;
par.msize = 8;
par.output = "mp4";
par.videoformat = 'mp4';
par.conn = [1 2; 1 3; 1 4; 3 6; 2 5; 3 6; 4 7; 5 8; 6 9; 9 12; 8 11; 7 10; 10 13; 13 16; 10 14; 10 15; 14 17; 15 18; 18 20; 17 19; 20 22; 19 21; 21 23; 22 24];
        
par2 = mcinitanimpar;
par2.msize = 8;
par2.output = "mp4";
par2.videoformat = 'mp4';
par2.conn = [1 2; 1 3; 1 4; 3 6; 2 5; 3 6; 4 7; 5 8; 6 9; 9 12; 8 11; 7 10; 10 13; 13 16; 10 14; 10 15; 14 17; 15 18; 18 20; 17 19; 20 22; 19 21; 21 23; 22 24];
     
trueD = mccenter(trueD);
predE = mccenter(predE);
      
[all, allparams] = mcmerge(trueD, mctranslate(predE, [2 0 0]), par, par2);
      
%Veri markers %%%%%Some files are mirrored in the true folder
par.markercolors='bgbbbbbbbbbbbbbbbbbbbbbg';
par2.markercolors='rgrrrrrrrrrrrrrrrrrrrrrg';

%true = "root", "lhip", "rhip", "belly", "lknee", "rknee",
%"lchest", "lankle", "rankle", "upchest", "ltoe", "rtoe", "neck",
%"lclavicle", "rclavicle", "head", "lshoulder", "rshoulder",
%"lelbow", "relbow", "lwrist", "rwrist", "lhand", "rhand"

        
%predicted = "root", "rhip", "lhip", "belly", "rknee", "lknee", 
%"lchest", "rankle", "lankle", "upchest", "rtoe", "ltoe", "neck", 
%"rclavicle", "lclavicle", "head", "rshoulder", "lshoulder", "relbow"
%"lelbow", "rwrist", "lwrist", "rhand", "lhand"
        
[all, allparams] = mcmerge(trueD, mctranslate(predE, [2 0 0]), par, par2);
mcplotframe(all, 1, allparams)
%End of veri markers

mirrored_files = [];
mirrored_files_index = [48, 49, 108, 109, 110, 111, 121, 128];
for i = mirrored_files_index
     mirrored_files = [mirrored_files, all_videos(i)];
end

%Fixing order of marker in mirrored files
new_order = [1 3 2 4 6 5 7 9 8 10 12 11 13 15 14 16 18 17 20 19 22 21 24 23];
dir_out = "/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/positions_amass_fixed_mirror/"
for k = 1:length(strList)
current_video = strList{k};
for i = mirrored_files
    is_mirrored = contains(current_video, i);
    if is_mirrored
        disp("found mirrored file")
        disp(i)
        disp(current_video)
        file_fix = strcat(true_files(1).folder, "/", current_video)
        
        True = readtable(file_fix);
        True = table2array(True);
        
        df = dance1;
        df.nFrames = height(True);
        df.nMarkers = width(True)/3;
        df.freq = 15;

        trueD = df;
        trueD.data = True;

        trueD.nMarkers=width(trueD.data)/3;
        trueD.nFrames = height(trueD.data);
        
        a = mcgetmarker(trueD, new_order)
        a = a.data
        
        fname_out = strcat(dir_out, current_video)
        
        writematrix(a, fname_out)
        
        break
    else
    end
end
end


