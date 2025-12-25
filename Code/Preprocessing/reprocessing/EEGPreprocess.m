% Start logging
diary("motor_imagery.txt")

% Load configuration files
data_root = 'D:\Code\Dataset_4-classification MI'; % Replace with your data root directory
save_root = fullfile(data_root, 'Data', 'Dataset v2.2');
info_file_path = fullfile(data_root, 'info.json');
channels_file_path = fullfile(data_root, 'channels.json');

% Read data information and channel-to-id mapping from JSON files
try
    info_content = fileread(info_file_path);
    info = jsondecode(info_content);
    channels_content = fileread(channels_file_path);
    channels = jsondecode(channels_content);
catch
    fprintf('Unable to load configuration files\n');
    return;
end

% Iterate through all time fields in the data structure
time_fields = fieldnames(info);
for i = 1:length(time_fields)
    field_name = time_fields{i};
    % Remove "x" prefix from field name
    if field_name(1) == 'x'
        time = field_name(2:end);
    else
        time = field_name;
    end
    data_groups = info.(field_name);
    
    % Iterate through each data group
    for j = 1:length(data_groups)
        data_group = data_groups(j);
        
        % Iterate through each data element
        for k = 1:length(data_group.data)
            item = data_group.data(k);
            subject_index = k;
            subject_id = sprintf('%02d', subject_index);
            filename = sprintf('subject_%s_motor_imagery_events.mat', subject_id);
            data_file_path = fullfile(data_root, 'Data', 'Dataset v1', time, 'EEG', filename);
            
            % Bad channels
            bads = item.bad_channels;
            bads_index = convertChannelsToIndices(bads, channels);
            fprintf('Time: %s, Data group: %d, Subject index: %d, Filename: %s\n', time, j, subject_index, filename);
            
            % Load data
            load(data_file_path)
            
            % Import data: 64 channels, sampling rate 250 Hz, epoch 25s
            EEG = pop_importdata('dataformat','array','nbchan',64,'data','data','srate',250,'pnts',[0 25],'xmin',0,'chanlocs','D:\Code\Dataset_4-classification MI\Data\Dataset v1\chanlocs.locs');
            
            % ICA decomposition with PCA reduction to 30 components
            EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'pca',30,'interrupt','on');
            
            % ICLabel artifact identification
            EEG = pop_iclabel(EEG, 'default');
            
            % Automatic screening of suspicious components
            % Brain: NaN NaN; Muscle: 0.9 1; Eye: 0.8 1
            % Heart: NaN NaN; Line Noise: NaN NaN; Channel Noise: 0.5 1; Other: NaN NaN
            EEG = pop_icflag(EEG, [NaN NaN;0.9 1;0.8 1;NaN NaN;NaN NaN;0.5 1;NaN NaN]);
            
            % Extract rejected component information
            rejected_info = displayRejectedComponents(EEG);
            disp(['Number of rejected components: ' num2str(rejected_info.count)]);
            
            % Remove rejected components
            EEG = pop_subcomp(EEG, [], 0);
            
            % Save preprocessed data
            saved_file_path = saveEEGData(EEG, save_root, time, filename, ch_names, label, bads, bads_index);
        end
    end
end

% End logging
diary off

function bads_index = convertChannelsToIndices(bads, channels)
% Convert channel names to corresponding indices for bad channel interpolation
% Input: bads - cell array of channel names; channels - structure mapping names to indices
% Output: bads_index - sorted array of channel indices

    if ~isempty(bads)
        bads_index = zeros(1, length(bads));
        
        for i = 1:length(bads)
            channel_name = bads{i};
            if isfield(channels, channel_name)
                bads_index(i) = channels.(channel_name);
            else
                warning('Channel %s not found in channels structure', channel_name);
                bads_index(i) = NaN;
            end
        end
        bads_index = bads_index(~isnan(bads_index));
        bads_index = sort(bads_index);
    else
        bads_index = [];
    end
end

function [rejected_info] = displayRejectedComponents(EEG)
% Display and analyze rejected ICA component information
% Input: EEG - EEGLAB EEG structure with ICA weights and IC flag information
% Output: rejected_info - structure containing rejected component details

    if ~isfield(EEG, 'reject') || ~isfield(EEG.reject, 'gcompreject')
        error('Input EEG data does not contain component rejection information (EEG.reject.gcompreject)');
    end
    
    if ~isfield(EEG, 'icaweights') || isempty(EEG.icaweights)
        error('Input EEG data does not contain ICA weight information');
    end
    
    rejected_comps = find(EEG.reject.gcompreject);
    
    rejected_info = struct();
    rejected_info.indices = rejected_comps;
    rejected_info.count = length(rejected_comps);
    rejected_info.percent = (rejected_info.count / size(EEG.icaweights, 1)) * 100;
    
    disp('=== Rejected Component Information ===');
    disp('Rejected component indices:');
    disp(rejected_comps');
    disp(['Total rejected components: ' num2str(rejected_info.count)]);
    disp(['Rejected component percentage: ' num2str(rejected_info.percent) '%']);
    
    if isfield(EEG.etc, 'ic_classification') && isfield(EEG.etc.ic_classification, 'ICLabel')
        classifications = EEG.etc.ic_classification.ICLabel.classifications;
        class_names = EEG.etc.ic_classification.ICLabel.classes;
        
        muscle_idx = find(strcmp(class_names, 'Muscle'));
        eye_idx = find(strcmp(class_names, 'Eye'));
        chanNoise_idx = find(strcmp(class_names, 'Channel Noise'));
        brain_idx = find(strcmp(class_names, 'Brain'));
        heart_idx = find(strcmp(class_names, 'Heart'));
        line_idx = find(strcmp(class_names, 'Line Noise'));
        other_idx = find(strcmp(class_names, 'Other'));
        
        disp('');
        disp('Rejected component classification details:');
        fprintf('Comp#  | Brain(N/A) | Muscle(>0.9) | Eye(>0.8) | Heart(N/A) | LineNoise(N/A) | ChanNoise(>0.5) | Other(N/A) | Rejection Reason\n');
        fprintf('----------------------------------------------------------------------------------------------------------\n');
        
        class_data = zeros(length(rejected_comps), 8);
        reasons = cell(length(rejected_comps), 1);
        
        for i = 1:length(rejected_comps)
            comp_idx = rejected_comps(i);
            
            brain_prob = classifications(comp_idx, brain_idx);
            muscle_prob = classifications(comp_idx, muscle_idx);
            eye_prob = classifications(comp_idx, eye_idx);
            heart_prob = classifications(comp_idx, heart_idx);
            line_prob = classifications(comp_idx, line_idx);
            chanNoise_prob = classifications(comp_idx, chanNoise_idx);
            other_prob = classifications(comp_idx, other_idx);
            
            reason = '';
            if muscle_prob >= 0.9
                reason = [reason 'Muscle '];
            end
            if eye_prob >= 0.8
                reason = [reason 'Eye '];
            end
            if chanNoise_prob >= 0.5
                reason = [reason 'Channel Noise '];
            end
            
            class_data(i, :) = [comp_idx, brain_prob, muscle_prob, eye_prob, heart_prob, line_prob, chanNoise_prob, other_prob];
            reasons{i} = reason;
            
            fprintf('#%3d   | %5.3f | %5.3f      | %5.3f   | %5.3f | %5.3f      | %5.3f         | %5.3f | %s\n', ...
                comp_idx, brain_prob, muscle_prob, eye_prob, heart_prob, line_prob, chanNoise_prob, other_prob, reason);
        end
        
        rejected_info.classification_table = array2table(class_data, ...
            'VariableNames', {'Component', 'Brain', 'Muscle', 'Eye', 'Heart', 'LineNoise', 'ChannelNoise', 'Other'});
        rejected_info.rejection_reasons = reasons;
        
        disp('');
        disp('Rejection reason statistics:');
        muscle_count = sum(class_data(:, 3) >= 0.9);
        eye_count = sum(class_data(:, 4) >= 0.8);
        channel_count = sum(class_data(:, 7) >= 0.5);
        
        fprintf('Muscle components (≥0.9): %d (%.1f%%)\n', muscle_count, (muscle_count/rejected_info.count)*100);
        fprintf('Eye components (≥0.8): %d (%.1f%%)\n', eye_count, (eye_count/rejected_info.count)*100);
        fprintf('Channel noise (≥0.5): %d (%.1f%%)\n', channel_count, (channel_count/rejected_info.count)*100);
        fprintf('Note: Some components may meet multiple rejection criteria\n');
    else
        disp('ICLabel classification information not found, cannot display detailed classification results.');
        rejected_info.classification_table = [];
        rejected_info.rejection_reasons = {};
    end
end

function saved_file_path = saveEEGData(EEG, save_root, time, filename, ch_names, label, bads, bad_index)
% Extract data from EEG structure and save as .mat file
% Input: EEG - EEGLAB EEG structure; save_root - save directory; time - time string
%        filename - save filename; ch_names - channel names; label - label data
%        bads - bad channel info; bad_index - bad channel indices
% Output: saved_file_path - full path of saved file

    preprocessed_data = EEG.data;
    srate = EEG.srate;

    save_path = fullfile(save_root, time, 'EEG');

    if ~exist(save_path, 'dir')
        mkdir(save_path);
    end

    if ~endsWith(filename, '.mat')
        filename = [filename, '.mat'];
    end

    full_filename = fullfile(save_path, filename);

    save(full_filename, 'preprocessed_data', 'srate', 'ch_names', 'label', 'bads', 'bad_index');

    saved_file_path = full_filename;
    
    disp(['Data successfully saved to: ', full_filename]);
end