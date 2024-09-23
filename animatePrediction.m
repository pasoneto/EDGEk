cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator/present/new2/'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox2')

load mcdemodata

%%read data
% directoryIn = "/Users/pdealcan/Documents/github/EDGEk/generatedDance/pairGenerated/"
% files = dir(directoryIn);

predicted_files_path = "/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/predicted_amass/";
pred_files = readtable("/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/random_selection_dances_perceptual_experiment.csv")
pred_files = pred_files.selected_dances;
true_files = dir("/Users/pdealcan/Documents/github/EDGEk/eval/eval_data/positions_amass/");
random_files = pred_files(randperm(length(pred_files)));

controls = [true false];
for c=1:2
    control = controls(c)
    for k=1:length(pred_files)
        nameChosen = pred_files(k);
        
        True = strcat(true_files(2).folder, "/", nameChosen);
        if control
           pred = strcat(predicted_files_path, "/", random_files(k));
        else
           pred = strcat(predicted_files_path, "/", nameChosen);
        end

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

        trueD = mccenter(trueD);
        predE = mccenter(predE);

        true_left = round(rand);

        if true_left == 1
            nameInit = "true_predicted"
            [all, allparams] = mcmerge(trueD, mctranslate(predE, [2 0 0]), par, par2);
        else
            [all, allparams] = mcmerge(predE, mctranslate(trueD, [2 0 0]), par2, par);
            nameInit = "predicted_true"
        end

        all.freq = 15;
        allparams.videoformat = "mp4";

        if control
            nameChosen = strcat("../videos_eval/control/", nameInit, "_control_", nameChosen);
        else
            nameChosen = strcat("../videos_eval/experiment/", nameInit, "_experiment_", nameChosen);
        end
mcanimate(all)
        allparams.output = strrep(nameChosen, "csv", "mp4");    
        mcanimate(all, allparams);

    end
end
