function [ data ]= trainingdata()
%% Training Examples
train = dir('train/*.txt');
iter = 1;
data = table();

% Open each text file and append its data to the table along with the
% corresponding output
for file = train'
    % The directory has 10 files
    % For each iter, store the file data in a temp variable
    temp = readtable(strcat('train/',file.name),...
    'Delimiter','space','ReadVariableNames',false);
   
    
    % Generate the output vector and convert type
    target = zeros(1,10);
    target(iter) = 1;
    target = ones(height(temp),1)*target;
    target = array2table(target,'VariableNames',...
    {'t0' 't1' 't2' 't3' 't4' 't5' 't6' 't7' 't8' 't9'});

    % Combine digit representations with their target output and 
    % append to the table
    data = [data; temp target];
    iter = iter + 1;
    
end

%% Clean data
% MATLAB(R) has useful functions for cleaning up tables. So we use them while
% data is in a table and then convert to a matrix for simplification when
% training the algorithm
missing = ismissing(data);
data = data(~any(missing,2),:);
data = table2array(data);

end
