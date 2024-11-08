cd '/Users/pdealcan/Documents/github/CoE_Neto/code/accelProject/danceGenerator/present/new2/'
addpath('/Users/pdealcan/Documents/github/matlabTools/MocapToolbox/mocaptoolbox2')

load mcdemodata

predicted_files = dir('../generated_dances/');
true_files = dir('../data/test/motions_sliced/');

% Define the list of marker names
list_of_names = {'root', 'rhip', 'lhip', 'belly', 'rknee', 'lknee', 'lchest', 'rankle', 'lankle', 'upchest', 'rtoe', 'ltoe', 'neck', 'rclavicle', 'lclavicle', 'head', 'rshoulder', 'lshoulder', 'relbow', 'lelbow', 'rwrist', 'lwrist', 'rhand', 'lhand'}
% Initialize an empty cell array to store the modified strings
markerNames = cell(1, numel(list_of_names) * 3);
for i = 1:numel(list_of_names)
    % Generate replicated strings with appended numbers
    for j = 1:3
        markerNames{(i-1)*3 + j} = [list_of_names{i}, num2str(j)];
    end
end

control = false;
all_files = [];
all_files_full = [];
for k=3:length(predicted_files)
  nameChosen = predicted_files(k).name;
    
  True = strcat(true_files(2).folder, '/', nameChosen);
  if control
      index = 0;
      index = randi([3, length(true_files)]);
      nameChosenRandom = true_files(index).name; 
      pred = strcat(predicted_files(2).folder, '/', nameChosenRandom);
  else
      pred = strcat(predicted_files(2).folder, '/', nameChosen);
  end
    
  True = readtable(True);
  pred = readtable(pred);

  True = table2array(True);
  pred = table2array(pred);
    
  % markers = ['root', 'lhip', 'rhip', 'belly', 'lknee', 'rknee', 'spine', 'lankle', 'rankle', 'chest', 'ltoes', 'rtoes', 'neck', 'linshoulder', 'rinshoulder', 'head',  'lshoulder', 'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 'lhand', 'rhand']
  %Adding to matlab object
  df = dance1;
  df.nFrames = height(True);
  df.nMarkers = width(True)/3;
  df.freq = 15;

  trueD = df;
  predE = df;
  diffs = df;
  
  trueD.data = True;
  predE.data = pred;
  trueD.nMarkers=width(trueD.data)/3;
  predE.nMarkers=width(predE.data)/3;
  diffs.nMarkers = width(predE.data)/3;

  trueD.nFrames = height(trueD.data);
  predE.nFrames = height(predE.data);
  diffs.nFrames = height(predE.data);

  trueD = mccenter(trueD);
  predE = mccenter(predE);
    
  %Mean Positional Error
  diffs.data = abs(trueD.data - predE.data);

  %diffs = array2table(diffs, 'VariableNames', markerNames);       
  m_for_markers = [];
  for l=1:24
      m_for_markers = [m_for_markers, mean(mcgetmarker(diffs, l).data, 'All')];
  end
    
  mpe_case = array2table(m_for_markers, 'VariableNames', list_of_names);

  %GTC    
  first_dimension_indexes = 1:3:size(trueD.data, 2);
  first_dimension = corr(mean(trueD.data(:, first_dimension_indexes), 2), mean(predE.data(:, first_dimension_indexes), 2));
  first_dimension_full_true = mean(trueD.data(:, first_dimension_indexes), 2);
  first_dimension_full_pred = mean(predE.data(:, first_dimension_indexes), 2);

  second_dimension_indexes = 2:3:size(trueD.data, 2);
  second_dimension = corr(mean(trueD.data(:, second_dimension_indexes), 2), mean(predE.data(:, second_dimension_indexes), 2));
  second_dimension_full_true = mean(trueD.data(:, second_dimension_indexes), 2);
  second_dimension_full_pred = mean(predE.data(:, second_dimension_indexes), 2);

  third_dimension_indexes = 3:3:size(trueD.data, 2);
  third_dimension = corr(mean(trueD.data(:, third_dimension_indexes), 2), mean(predE.data(:, third_dimension_indexes), 2));      
  third_dimension_full_true = mean(trueD.data(:, third_dimension_indexes), 2);
  third_dimension_full_pred = mean(predE.data(:, third_dimension_indexes), 2);
    
  %gtc_complete = array2table([first_dimension_full_true second_dimension_full_true third_dimension_full_true first_dimension_full_pred second_dimension_full_pred third_dimension_full_pred], 'VariableNames', ["x_true", "y_true", "z_true", "x_pred", "y_pred", "z_pred"]);
  %gtc_complete.file = repmat(k, 150, 1);
    
  gtc_dimensions = ["gtc_first", "gtc_second", "gtc_third"];
  gtc_case = array2table([first_dimension, second_dimension, third_dimension], 'VariableNames', gtc_dimensions);
    
  gtc_case.file = string(nameChosen);
  if control
      gtc_case.condition = "control";
  else
      gtc_case.condition = "experiment";
  end

  objective_measure_case = [mpe_case, gtc_case];
  all_files = [all_files, {objective_measure_case}];
  %all_files_full = [all_files_full, {gtc_complete}];

end

combined_table = [all_files{1}]
for i = 2:length(all_files)
    combined_table = vertcat(combined_table, all_files{i});
end

%combined_table2 = [all_files_full{1}]
%for i = 2:length(all_files_full)
    %combined_table2 = vertcat(combined_table2, all_files_full{i});
%end

if control
    nOut = strcat("./eval_data/objective_measures/", "objective_measure_control.csv")
else
    nOut = strcat("./eval_data/objective_measures/", "objective_measure_experimental.csv")
end

%writetable(combined_table2, "./eval_data/objective_measures/gtc_full.csv")

% writetable(combined_table, nOut)
