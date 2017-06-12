function [ dummy ]= testingdata()
%% Testing Examples
test = dir('test/*.txt');
iter = 1;
dummy = table();

% Open each text file and append its data to the table along with the
% corresponding output
for file = test'
    % 10 test files: 1 for each digit from 0-9
    % For each iter, store the file data in a dummy variable
    dummy = readtable(strcat('test/',file.name),...
    'Delimiter','space','ReadVariableNames',false);
   
    % output label vector
    label = (ones(1).*-1);
    label(0) = 1;
    label = ones(height(dummy),1)*label;
    label = array2table(label,'VariableNames',...
    {'t0'});

    % Combine digit representations with their output label and 
    % append to the table
    dummy = [dummy; dummy label];
    iter = iter + 1;
    
    
end

%% Cleaning data and converting to matrix for data manipulation.

missing = ismissing(dummy);
dummy = dummy(~any(missing,2),:);
dummy = table2array(dummy);

end
