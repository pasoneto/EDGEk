cd '/Users/pdealcan/Documents/github/jatosTEST/study_assets_root/MMBB/videos/dance_poser/singles'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox2')

load mcdemodata

predicted_files_path = "/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/predicted_amass/";
pred_files = dir(predicted_files_path);

true_files = dir("/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/positions_amass/");

filtered_pred_files = {};
% Loop through each substring and check if it is present in the main string
for l=3:length(pred_files)
    substrings = {'sliced_0_', 'Angry', 'Curiosity', 'Happy', 'Nervous', 'Sad', 'Scary', 'Excited', 'Annoyed', 'Bored', 'Miserable', 'Mix', 'Pleased', 'Relaxed', 'Sad', 'Satisfied', 'Tired', 'Afraid', 'Neutral'};
    cName = pred_files(l).name;
    dance_slice = true;
    for i = 1:length(substrings)
        if contains(cName, substrings{i})
            dance_slice = false;
            break; % Exit the loop if any substring is found
        else
            disp("Still dance")
        end
    end
    if dance_slice
        filtered_pred_files = [filtered_pred_files cName];
    else
        disp("Filtered out")
    end
end

pred_files = filtered_pred_files;

for k=147:length(pred_files)
    nameChosen = pred_files(k);
        
    True = strcat(true_files(2).folder, "/", nameChosen);
    pred = strcat(predicted_files_path, nameChosen);

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
%     par.freq = 15;
    par.videoformat = "mp4";
    
    par2 = mcinitanimpar
    par2.msize = 8
    par2.output = "mp4";
    par2.videoformat = 'mp4'
    par2.conn = [1 2; 1 3; 1 4; 3 6; 2 5; 3 6; 4 7; 5 8; 6 9; 9 12; 8 11; 7 10; 10 13; 13 16; 10 14; 10 15; 14 17; 15 18; 18 20; 17 19; 20 22; 19 21; 21 23; 22 24];
    par2.markercolors='rrrrrrrrrrrrrrrrrrrrrrrr'
%     par2.freq = 15;
    par2.videoformat = "mp4";

    trueD = mccenter(trueD);
    predE = mccenter(predE);
    
    nameChosenTrue = strcat('/Users/pdealcan/Documents/github/jatosTEST/study_assets_root/MMBB/videos/dance_poser/singles/control/', 'true_', nameChosen);
    nameChosenPred = strcat('/Users/pdealcan/Documents/github/jatosTEST/study_assets_root/MMBB/videos/dance_poser/singles/experiment/', 'pred_', nameChosen);

    red_or_blue = round(rand);
    if red_or_blue == 1
        disp("Got here 1a")
        par.output = char(strrep(nameChosenTrue, 'csv', 'mp4')); 
        disp("Got here 2a")
        par2.output = char(strrep(nameChosenPred, 'csv', 'mp4')); 
        disp("Got here 3a")
        mcanimate(trueD, par);
        disp("Got here 4a")
        mcanimate(predE, par2);
        disp("Got here 5a")
    else
        disp("Got here 1b")
        par.output = char(strrep(nameChosenPred, 'csv', 'mp4')); 
        disp("Got here 2b")
        par2.output = char(strrep(nameChosenTrue, 'csv', 'mp4')); 
        disp("Got here 3b")
        mcanimate(trueD, par2);
        disp("Got here 4b")
        mcanimate(predE, par);
        disp("Got here 5b")
    end
end

