%% Multimodal EEG and NIRS Classification Fusion
% This code combines EEG and NIRS data processing and classification techniques
% for multimodal fusion classification of motor imagery tasks.

%% Initialize Variables and Data Paths
clear all; close all;

% Load configuration file
data_root = 'D:\Code\Dataset_4-classification MI'; % Replace with your data root directory
info_file_path = fullfile(data_root, 'info.json');

% Working directories - Update to your local paths
EegMyDataDir = 'D:\Code\Dataset_4-classification MI\Data\Dataset v2';
TemDir = 'D:\Code\Dataset_4-classification MI\Data\Dataset v2';
WorkingDir = 'D:\Code\Dataset_4-classification MI\CSP-LDA';

% Read data information and channel-ID mappings from JSON file
try
    info_content = fileread(info_file_path);
    info = jsondecode(info_content);
catch
    fprintf('Unable to load configuration file\n');
    return;
end

%% Iterate Through All Time Sessions in Data Structure
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
            
            % Subject index
            subject_index = k; % MATLAB index starts from 1
            subject_id = sprintf('%02d', subject_index); % Format as two digits
            
            % Construct file names using sprintf
            filename = sprintf('subject_%s_motor_imagery_events.mat', subject_id);
            nirs_folder_name = sprintf('sub_%s', subject_id);
            
            % Data directories
            eeg_data_path = fullfile(data_root, 'Data', 'Dataset v2', time, 'EEG', filename);
            hbo_filepath = fullfile(data_root, 'Data', 'Dataset v2', time, 'NIRS', nirs_folder_name, 'hbo.csv');
            hbr_filepath = fullfile(data_root, 'Data', 'Dataset v2', time, 'NIRS', nirs_folder_name, 'hbr.csv');
            
            % Check if files exist
            if ~exist(hbo_filepath, 'file') || ~exist(hbr_filepath, 'file')
                fprintf('Files not found, skipping: %s\n', nirs_folder_name);
                continue;
            end
            
            %% Load Data
            % Load EEG data
            load(eeg_data_path);
            disp(['EEG data path:', eeg_data_path]);
            
            % Load NIRS data
            % [120, 51, 301] represents 120 events, 51 channels, 301 sample points (-5 to 25 seconds)
            opts = struct('skipFirstRow', true, 'dimensions', [120, 51, 301], 'verbose', true);
            [hbo_fnirs_data, nirs_labels] = loadFnirsData(hbo_filepath, opts);
            [hbr_fnirs_data, ~] = loadFnirsData(hbr_filepath, opts);
            disp(['HbO data path:', hbo_filepath]);
            disp(['HbR data path:', hbr_filepath]);
            
            %% Manual EEG Data Processing
            % Create EEG cnt structure
            eeg_cnt = struct();
            
            % Ensure x field is double precision floating point matrix [timepoints × channels]
            eeg_cnt.x = [];
            for m = 1:size(preprocessed_data, 3)
                event_data = preprocessed_data(:, :, m);
                eeg_cnt.x = [eeg_cnt.x; double(event_data')];
            end
            
            % Ensure clab is a cell array of strings
            if ~iscell(ch_names)
                clab_cell = cell(1, length(ch_names));
                for n = 1:length(ch_names)
                    if ischar(ch_names(n)) || isstring(ch_names(n))
                        clab_cell{n} = char(ch_names(n));
                    else
                        clab_cell{n} = sprintf('Ch%d', n);
                    end
                end
                eeg_cnt.clab = clab_cell;
            else
                eeg_cnt.clab = ch_names;
            end
            
            % Set sampling rate
            eeg_cnt.fs = double(srate);
            
            % Create EEG mrk structure
            eeg_mrk = struct();
            
            % Set time field as row vector
            si = 1000/eeg_cnt.fs;
            event_length = 6250 * si;
            eeg_mrk.time = zeros(1, 120);
            for m = 1:120
                eeg_mrk.time(m) = 1250 * si + (m-1) * event_length;
            end
            
            % Ensure y is in correct format (one-hot encoded logical matrix)
            eeg_mrk.y = logical(label);
            
            % Set class names
            eeg_mrk.className = {'Left to right', 'Top to bottom', 'Top left to bottom right', 'Top right to bottom left'};
            
            %% Manual NIRS Data Loading and Preprocessing
            fprintf('Processing NIRS data for subject %s\n', subject_id);
            
            % Convert NIRS data to BBCI toolbox format
            [hbo_cnt, nirs_mrk] = convertNirsToBBCItoolbox(hbo_fnirs_data, nirs_labels);
            [hbr_cnt, ~] = convertNirsToBBCItoolbox(hbr_fnirs_data, nirs_labels);
            
            %% Initialize BBCI Toolbox
            startup_bbci_toolbox('DataDir', EegMyDataDir, 'TmpDir', TemDir);
            BTB.History = 0; % Avoid errors when merging cnt
            
            %% Define Channel Groups for Both Modalities
            % EEG channels by brain region
            MotorChannel_EEG = {'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', ...
                'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', ...
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'};
            
            ParietalChannel_EEG = {'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'};
            
            FrontalChannel_EEG = {'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', ...
                'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'};
            
            OccipitalChannel_EEG = {'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', ...
                'CB1', 'O1', 'OZ', 'O2', 'CB2'};
            
            TemporalChannel_EEG = {'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'M1', 'M2'};
            
            % NIRS channels by brain region
            FrontalChannel_NIRS = {'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', ...      % Left frontal
                'Ch35', 'Ch36', 'Ch37', 'Ch38', 'Ch39', 'Ch40', 'Ch41', ... % Central frontal
                'Ch18', 'Ch19', 'Ch20', 'Ch21', 'Ch22', 'Ch23', 'Ch24'};    % Right frontal
            
            MotorChannel_NIRS = {'Ch9', 'Ch10', 'Ch12', ...                % Left motor
                'Ch42', 'Ch43', 'Ch44', 'Ch45', ...                        % Central motor
                'Ch26', 'Ch27', 'Ch29'};                                  % Right motor
            
            ParietalChannel_NIRS = {'Ch13', 'Ch14', 'Ch15', 'Ch16', ...    % Left parietal
                'Ch46', 'Ch47', 'Ch48', 'Ch49', 'Ch50', 'Ch51', ...        % Central parietal
                'Ch30', 'Ch31', 'Ch32', 'Ch33'};                          % Right parietal
            
            %% Channel Selection
            % Select EEG channels
            eeg_cnt_org = eeg_cnt; % Backup
            eeg_cnt = proc_selectChannels(eeg_cnt, [MotorChannel_EEG, ParietalChannel_EEG]);
            
            % Select NIRS channels
            hbo_cnt_org = hbo_cnt; % Backup
            hbr_cnt_org = hbr_cnt; % Backup
            hbo_cnt = proc_selectChannels(hbo_cnt, [FrontalChannel_NIRS, MotorChannel_NIRS, ParietalChannel_NIRS]);
            hbr_cnt = proc_selectChannels(hbr_cnt, [FrontalChannel_NIRS, MotorChannel_NIRS, ParietalChannel_NIRS]);
            
            %% Segmentation
            % EEG segmentation
            eeg_ival_epo = [-5000 20000];  % Time window range (milliseconds)
            eeg_epo = proc_segmentation(eeg_cnt, eeg_mrk, eeg_ival_epo);
            
            %% Reorder EEG Epochs
            % For each block of 40 epochs, reorder so that all class 1 appears first,
            % then class 2, class 3, class 4, while maintaining original order within each class
            num_blocks = 3;  % Total 120 epochs, 40 per block
            epochs_per_block = 40;
            num_classes = 4;
            
            % Create temporary storage for reordered data
            temp_x = zeros(size(eeg_epo.x));
            temp_y = false(size(eeg_epo.y));
            
            % Process each block of 40 epochs
            for block = 1:num_blocks
                block_start = (block-1) * epochs_per_block + 1;
                block_end = block * epochs_per_block;
                
                % Initialize index counter for new positions in reordered block
                new_idx = block_start;
                
                % Process each class
                for class = 1:num_classes
                    % Find indices of current class in current block
                    block_class_indices = find(eeg_epo.y(class, block_start:block_end));
                    
                    % Adjust indices to absolute positions in original data
                    orig_indices = block_start + block_class_indices - 1;
                    
                    % Copy data of this class to new positions
                    for m = 1:length(orig_indices)
                        orig_idx = orig_indices(m);
                        temp_x(:,:,new_idx) = eeg_epo.x(:,:,orig_idx);
                        temp_y(:,new_idx) = eeg_epo.y(:,orig_idx);
                        new_idx = new_idx + 1;
                    end
                end
            end
            
            % Replace original data with reordered data
            eeg_epo.x = temp_x;
            eeg_epo.y = temp_y;
            
            %% NIRS Segmentation
            nirs_ival_epo = [-5000 20000];  % Time window range (milliseconds)
            nirs_epo.oxy = proc_segmentation(hbo_cnt, nirs_mrk, nirs_ival_epo);
            nirs_epo.deoxy = proc_segmentation(hbr_cnt, nirs_mrk, nirs_ival_epo);
            
            %% EEG Processing - Frequency Band Selection
            % Define filter bank frequency bands (Hz)
            nBands = 9;
            freqBands = zeros(nBands, 2);
            for m = 1:nBands
                freqBands(m,:) = [4+(m-1)*4, 8+(m-1)*4];
            end
            
            %% Time Window Parameters for Both Modalities
            StepSize = 1*1000; % milliseconds
            WindowSize = 3*1000; % milliseconds
            
            % EEG time windows
            eeg_ival_start = (eeg_ival_epo(1):StepSize:eeg_ival_epo(end)-WindowSize)';
            eeg_ival_end = eeg_ival_start+WindowSize;
            eeg_ival = [eeg_ival_start, eeg_ival_end];
            eeg_nStep = length(eeg_ival);
            
            % NIRS time windows
            nirs_ival_start = (nirs_ival_epo(1):StepSize:nirs_ival_epo(end)-WindowSize)';
            nirs_ival_end = nirs_ival_start+WindowSize;
            nirs_ival = [nirs_ival_start, nirs_ival_end];
            nirs_nStep = length(nirs_ival);
            
            % Select moving time windows for EEG
            for stepIdx = 1:eeg_nStep
                eeg_segment{stepIdx} = proc_selectIval(eeg_epo, eeg_ival(stepIdx,:));
            end
            
            % Process NIRS data, calculate mean and slope
            for stepIdx = 1:nirs_nStep
                % Mean calculation
                nirs_ave.deoxy{stepIdx} = proc_meanAcrossTime(nirs_epo.deoxy, nirs_ival(stepIdx,:));
                nirs_ave.oxy{stepIdx} = proc_meanAcrossTime(nirs_epo.oxy, nirs_ival(stepIdx,:));
                
                % Slope calculation
                nirs_slope.deoxy{stepIdx} = proc_slopeAcrossTime(nirs_epo.deoxy, nirs_ival(stepIdx,:));
                nirs_slope.oxy{stepIdx} = proc_slopeAcrossTime(nirs_epo.oxy, nirs_ival(stepIdx,:));
            end
            
            %% Cross-Validation Setup
            nShift = 2; % Number of repetitions
            nFold = 5;  % Number of folds
            
            % Get group labels
            eeg_group = eeg_epo.y;
            nirs_group = nirs_epo.deoxy.y; % epo.deoxy.y == epo.oxy.y
            
            % Number of classes (should be 4 for four-class)
            nClasses = size(eeg_group, 1);
            
            % Number of CSP filters selected per frequency band
            nFilters = 2; % Filters per class per band (top nFilters and bottom nFilters)
            
            % Initialize accuracy metrics
            acc.eeg = zeros(nShift, eeg_nStep);
            acc.nirs_deoxy = zeros(nShift, nirs_nStep);
            acc.nirs_oxy = zeros(nShift, nirs_nStep);
            acc.fusion = zeros(nShift, min(eeg_nStep, nirs_nStep));
            
            % Initialize confusion matrices
            cmat.eeg = zeros(nClasses, nClasses, nFold);
            cmat.nirs_deoxy = zeros(nClasses, nClasses, nFold);
            cmat.nirs_oxy = zeros(nClasses, nClasses, nFold);
            cmat.fusion = zeros(nClasses, nClasses, nFold);
            
            % Initialize prediction results
            grouphat = struct();
            
            %% Cross-Validation and Multimodal Fusion
            for shiftIdx = 1:nShift
                % Create cross-validation indices
                indices{shiftIdx} = crossvalind('Kfold', full(vec2ind(eeg_group)), nFold);
                
                for stepIdx = 1:min(eeg_nStep, nirs_nStep)
                    fprintf('Multimodal fusion, Repeat:%d/%d, Step:%d/%d\n', shiftIdx, nShift, stepIdx, min(eeg_nStep, nirs_nStep));
                    
                    % Pre-compute filtered data for all frequency bands outside CV loop
                    filtered_data_all_bands = cell(nBands, 1);
                    eeg_all_data = eeg_segment{stepIdx};
                    
                    for bandIdx = 1:nBands
                        filtered_data_all_bands{bandIdx} = applyBandpassFilter(eeg_all_data, freqBands(bandIdx,1), freqBands(bandIdx,2));
                    end
                    
                    for foldIdx = 1:nFold
                        % Split train and test sets
                        test = (indices{shiftIdx} == foldIdx);
                        train = ~test;
                        y_test = vec2ind(eeg_group(:,test));
                        
                        %% EEG Processing
                        % Initialize classifiers
                        eeg_C = cell(nClasses, 1);
                        eeg_fv_train = cell(nClasses, 1);
                        eeg_fv_test = cell(nClasses, 1);
                        
                        for classIdx = 1:nClasses
                            % Create binary classification labels (one-vs-rest)
                            curr_class_idx = find(eeg_group(classIdx,train) == 1);
                            other_class_idx = find(eeg_group(classIdx,train) ~= 1);
                            
                            binary_labels = zeros(2, length(curr_class_idx)+length(other_class_idx));
                            binary_labels(1, curr_class_idx) = 1;  % Current class
                            binary_labels(2, other_class_idx) = 1; % All other classes
                            
                            % FBCSP: Process all frequency bands
                            fv_train_allbands = [];
                            fv_test_allbands = [];
                            
                            for bandIdx = 1:nBands
                                % Use pre-filtered data
                                all_data_band = filtered_data_all_bands{bandIdx};
                                
                                % Separate train and test sets
                                x_train_band = struct();
                                x_train_band.x = all_data_band.x(:,:,train);
                                x_train_band.y = all_data_band.y(:,train);
                                x_train_band.clab = all_data_band.clab;
                                
                                x_test_band = struct();
                                x_test_band.x = all_data_band.x(:,:,test);
                                x_test_band.y = all_data_band.y(:,test);
                                x_test_band.clab = all_data_band.clab;
                                
                                % Setup binary classification data
                                x_train_binary = x_train_band;
                                third_dimension_size = size(x_train_binary.x, 3);
                                x_train_binary.y = binary_labels(:,1:third_dimension_size);
                                
                                % Apply CSP algorithm (compute CSP matrix using training data only)
                                [csp_train, CSP_W, CSP_EIG, CSP_A] = proc_cspAuto(x_train_binary);
                                
                                % Select first 2 and last 2 CSP features
                                csp_train.x = csp_train.x(:,[1 2 end-1 end],:);
                                
                                % Apply same CSP matrix to test data
                                csp_test = struct();
                                for testIdx = 1:size(find(test==1),1)
                                    csp_test.x(:,:,testIdx) = x_test_band.x(:,:,testIdx)*CSP_W;
                                end
                                csp_test.x = csp_test.x(:,[1 2 end-1 end],:);
                                csp_test.y = x_test_band.y;
                                csp_test.clab = x_test_band.clab;
                                
                                % Calculate features for train and test sets
                                var_train = proc_variance(csp_train);
                                logvar_train = proc_logarithm(var_train);
                                var_test = proc_variance(csp_test);
                                logvar_test = proc_logarithm(var_test);
                                
                                % Concatenate features from all frequency bands
                                if isempty(fv_train_allbands)
                                    fv_train_allbands = squeeze(logvar_train.x);
                                else
                                    fv_train_allbands = [fv_train_allbands; squeeze(logvar_train.x)];
                                end
                                
                                if isempty(fv_test_allbands)
                                    fv_test_allbands = squeeze(logvar_test.x);
                                else
                                    fv_test_allbands = [fv_test_allbands; squeeze(logvar_test.x)];
                                end
                            end
                            
                            % Store training and testing data
                            eeg_fv_train{classIdx} = fv_train_allbands;
                            eeg_fv_test{classIdx} = fv_test_allbands;
                            
                            % Train classifier
                            eeg_C{classIdx} = train_RLDAshrink(fv_train_allbands, binary_labels);
                        end
                        
                        %% NIRS Processing
                        % Initialize NIRS classifiers
                        nirs_C.deoxy = cell(nClasses, 1);
                        nirs_C.oxy = cell(nClasses, 1);
                        
                        % Prepare NIRS training and testing data
                        x_train.deoxy.x = [squeeze(nirs_ave.deoxy{stepIdx}.x(:,:,train)); squeeze(nirs_slope.deoxy{stepIdx}.x(:,:,train))];
                        x_train.deoxy.y = squeeze(nirs_group(:,train));
                        x_train.deoxy.clab = nirs_ave.deoxy{stepIdx}.clab;
                        
                        x_train.oxy.x = [squeeze(nirs_ave.oxy{stepIdx}.x(:,:,train)); squeeze(nirs_slope.oxy{stepIdx}.x(:,:,train))];
                        x_train.oxy.y = squeeze(nirs_group(:,train));
                        x_train.oxy.clab = nirs_ave.oxy{stepIdx}.clab;
                        
                        x_test.deoxy.x = [squeeze(nirs_ave.deoxy{stepIdx}.x(:,:,test)); squeeze(nirs_slope.deoxy{stepIdx}.x(:,:,test))];
                        x_test.deoxy.y = squeeze(nirs_group(:,test));
                        x_test.deoxy.clab = nirs_ave.deoxy{stepIdx}.clab;
                        
                        x_test.oxy.x = [squeeze(nirs_ave.oxy{stepIdx}.x(:,:,test)); squeeze(nirs_slope.oxy{stepIdx}.x(:,:,test))];
                        x_test.oxy.y = squeeze(nirs_group(:,test));
                        x_test.oxy.clab = nirs_ave.oxy{stepIdx}.clab;
                        
                        % Train NIRS classifiers for each class
                        for classIdx = 1:nClasses
                            % Create binary classification labels
                            curr_class_idx = find(nirs_group(classIdx,train) == 1);
                            other_class_idx = find(nirs_group(classIdx,train) ~= 1);
                            
                            binary_labels = zeros(2, length(curr_class_idx)+length(other_class_idx));
                            binary_labels(1, curr_class_idx) = 1;  % Current class
                            binary_labels(2, other_class_idx) = 1; % All other classes
                            
                            % Train deoxy classifier
                            fv_train_deoxy = struct();
                            fv_train_deoxy.x = x_train.deoxy.x;
                            second_dimension_size = size(fv_train_deoxy.x, 2);
                            fv_train_deoxy.y = binary_labels(:,1:second_dimension_size);
                            nirs_C.deoxy{classIdx} = train_RLDAshrink(fv_train_deoxy.x, fv_train_deoxy.y);
                            
                            % Train oxy classifier
                            fv_train_oxy = struct();
                            fv_train_oxy.x = x_train.oxy.x;
                            fv_train_oxy.y = binary_labels(:,1:second_dimension_size);
                            nirs_C.oxy{classIdx} = train_RLDAshrink(fv_train_oxy.x, fv_train_oxy.y);
                        end
                        
                        %% Apply Base Classifiers to Get Decision Values
                        % Initialize classifier outputs
                        out_train = struct();
                        out_test = struct();
                        
                        % Apply EEG classifiers
                        out_train.eeg = zeros(nClasses, size(find(train==1), 1));
                        out_test.eeg = zeros(nClasses, size(find(test==1), 1));
                        
                        for classIdx = 1:nClasses
                            [~, out_test.eeg(classIdx,:)] = applyRLDAClassifier(eeg_fv_test{classIdx}, eeg_C{classIdx});
                            [~, out_train.eeg(classIdx,:)] = applyRLDAClassifier(eeg_fv_train{classIdx}, eeg_C{classIdx});
                        end
                        
                        % Apply NIRS classifiers
                        out_train.deoxy = zeros(nClasses, size(find(train==1), 1));
                        out_train.oxy = zeros(nClasses, size(find(train==1), 1));
                        out_test.deoxy = zeros(nClasses, size(find(test==1), 1));
                        out_test.oxy = zeros(nClasses, size(find(test==1), 1));
                        
                        for classIdx = 1:nClasses
                            [~, out_test.deoxy(classIdx,:)] = applyRLDAClassifier(x_test.deoxy.x, nirs_C.deoxy{classIdx});
                            [~, out_test.oxy(classIdx,:)] = applyRLDAClassifier(x_test.oxy.x, nirs_C.oxy{classIdx});
                            [~, out_train.deoxy(classIdx,:)] = applyRLDAClassifier(x_train.deoxy.x, nirs_C.deoxy{classIdx});
                            [~, out_train.oxy(classIdx,:)] = applyRLDAClassifier(x_train.oxy.x, nirs_C.oxy{classIdx});
                        end
                        
                        %% Feature Fusion - Create Meta-Features
                        % Initialize meta-features for test data
                        meta_features = struct();
                        
                        % Create different combinations of modalities
                        meta_features.meta1 = [out_test.deoxy; out_test.oxy];           % HbR + HbO
                        meta_features.meta2 = [out_test.deoxy; out_test.eeg];           % HbR + EEG
                        meta_features.meta3 = [out_test.oxy; out_test.eeg];             % HbO + EEG
                        meta_features.meta4 = [out_test.deoxy; out_test.oxy; out_test.eeg]; % HbR + HbO + EEG
                        
                        %% Train Meta-Classifiers Using One-vs-Rest Approach
                        % Initialize meta-classifiers
                        meta_C = struct();
                        meta_C.meta1 = cell(nClasses, 1);
                        meta_C.meta2 = cell(nClasses, 1);
                        meta_C.meta3 = cell(nClasses, 1);
                        meta_C.meta4 = cell(nClasses, 1);
                        
                        % Create meta-features for training
                        meta_train = struct();
                        meta_train.meta1 = [out_train.deoxy; out_train.oxy];
                        meta_train.meta2 = [out_train.deoxy; out_train.eeg];
                        meta_train.meta3 = [out_train.oxy; out_train.eeg];
                        meta_train.meta4 = [out_train.deoxy; out_train.oxy; out_train.eeg];
                        
                        % Train meta-classifiers for each class using one-vs-rest
                        for classIdx = 1:nClasses
                            % Create binary classification labels
                            curr_class_idx = find(nirs_group(classIdx,train) == 1);
                            other_class_idx = find(nirs_group(classIdx,train) ~= 1);
                            
                            binary_labels = zeros(2, length(curr_class_idx)+length(other_class_idx));
                            binary_labels(1, curr_class_idx) = 1;  % Current class
                            binary_labels(2, other_class_idx) = 1; % All other classes
                            
                            % Train meta-classifiers for different feature combinations
                            meta_C.meta1{classIdx} = train_RLDAshrink(meta_train.meta1, binary_labels);
                            meta_C.meta2{classIdx} = train_RLDAshrink(meta_train.meta2, binary_labels);
                            meta_C.meta3{classIdx} = train_RLDAshrink(meta_train.meta3, binary_labels);
                            meta_C.meta4{classIdx} = train_RLDAshrink(meta_train.meta4, binary_labels);
                        end
                        
                        %% Apply Meta-Classifiers for Final Prediction
                        meta_out = struct();
                        meta_out.meta1 = zeros(nClasses, size(find(test==1), 1));
                        meta_out.meta2 = zeros(nClasses, size(find(test==1), 1));
                        meta_out.meta3 = zeros(nClasses, size(find(test==1), 1));
                        meta_out.meta4 = zeros(nClasses, size(find(test==1), 1));
                        
                        for classIdx = 1:nClasses
                            [~, meta_out.meta1(classIdx,:)] = applyRLDAClassifier(meta_features.meta1, meta_C.meta1{classIdx});
                            [~, meta_out.meta2(classIdx,:)] = applyRLDAClassifier(meta_features.meta2, meta_C.meta2{classIdx});
                            [~, meta_out.meta3(classIdx,:)] = applyRLDAClassifier(meta_features.meta3, meta_C.meta3{classIdx});
                            [~, meta_out.meta4(classIdx,:)] = applyRLDAClassifier(meta_features.meta4, meta_C.meta4{classIdx});
                        end
                        
                        % Final decision - Choose class with minimum score for each sample
                        [~, pred_meta1] = min(meta_out.meta1, [], 1);
                        [~, pred_meta2] = min(meta_out.meta2, [], 1);
                        [~, pred_meta3] = min(meta_out.meta3, [], 1);
                        [~, pred_meta4] = min(meta_out.meta4, [], 1);
                        
                        grouphat.meta1{foldIdx} = pred_meta1;
                        grouphat.meta2{foldIdx} = pred_meta2;
                        grouphat.meta3{foldIdx} = pred_meta3;
                        grouphat.meta4{foldIdx} = pred_meta4;
                        
                        % Calculate confusion matrices
                        cmat.meta1(:,:,foldIdx) = confusionmat(y_test, grouphat.meta1{foldIdx});
                        cmat.meta2(:,:,foldIdx) = confusionmat(y_test, grouphat.meta2{foldIdx});
                        cmat.meta3(:,:,foldIdx) = confusionmat(y_test, grouphat.meta3{foldIdx});
                        cmat.meta4(:,:,foldIdx) = confusionmat(y_test, grouphat.meta4{foldIdx});
                    end
                    
                    % Calculate accuracy for each modality and fusion
                    acc.meta1(shiftIdx,stepIdx) = trace(sum(cmat.meta1,3)) / sum(sum(sum(cmat.meta1,3)));
                    acc.meta2(shiftIdx,stepIdx) = trace(sum(cmat.meta2,3)) / sum(sum(sum(cmat.meta2,3)));
                    acc.meta3(shiftIdx,stepIdx) = trace(sum(cmat.meta3,3)) / sum(sum(sum(cmat.meta3,3)));
                    acc.meta4(shiftIdx,stepIdx) = trace(sum(cmat.meta4,3)) / sum(sum(sum(cmat.meta4,3)));
                end
            end
            
            % Calculate mean accuracy across repetitions
            mean_acc.meta1 = mean(acc.meta1,1)';
            mean_acc.meta2 = mean(acc.meta2,1)';
            mean_acc.meta3 = mean(acc.meta3,1)';
            mean_acc.meta4 = mean(acc.meta4,1)';
            
            % Save results to file
            result = [mean_acc.meta1, mean_acc.meta2, mean_acc.meta3, mean_acc.meta4];
            
            % Visualization
            T = (eeg_ival(:,2)/1000)';
            fig = figure();
            plot(T, result);
            legend('HbR + HbO', 'HbR + EEG', 'HbO + EEG', 'HbR + HbO + EEG');
            xlim([T(1) T(end)]); ylim([0 0.6]); grid on;
            title(sprintf('%s Subject %s - Multimodal Classification', time, subject_id));
            xlabel('Time (s)');
            ylabel('Classification Accuracy');
            
            % Save figure
            filename = sprintf('%s_subject_%s_multimodal.fig', time, subject_id);
            disp(['Filename:', filename]);
            saveas(fig, filename);  % Save as .fig format
            
            % Close figure
            close(fig);
        end
    end
end

%% Auxiliary Functions
% Function to load fNIRS data
function [fnirs_data, labels] = loadFnirsData(filepath, options)
% LOADFNIRSDATA Load and process fNIRS data
%   [FNIRS_DATA, LABELS] = LOADFNIRSDATA(FILEPATH) loads fNIRS data from specified CSV file
%
%   [FNIRS_DATA, LABELS] = LOADFNIRSDATA(FILEPATH, OPTIONS) processes with specified options
%
%   Input arguments:
%       FILEPATH - String, CSV file path
%       OPTIONS - Structure with optional fields:
%           .skipFirstRow - Boolean, whether to skip first row (default: auto-detect)
%           .dimensions - 3-element vector, reshape dimensions (default: [120, 51, 301])
%           .verbose - Boolean, whether to print information (default: true)

    % Check if file exists
    if ~isfile(filepath)
        error('File not found: %s', filepath);
    end
    
    % Set default options
    defaultOpts = struct(...
        'skipFirstRow', [], ...  % Auto-detect
        'dimensions', [120, 51, 301], ...
        'verbose', true ...
    );
    
    % Process input options
    if nargin < 2
        options = defaultOpts;
    else
        % Apply default values for any missing fields
        fields = fieldnames(defaultOpts);
        for i = 1:length(fields)
            if ~isfield(options, fields{i})
                options.(fields{i}) = defaultOpts.(fields{i});
            end
        end
    end
    
    % Read CSV file
    try
        data = readmatrix(filepath);
    catch e
        error('Error reading %s:\n%s', filepath, e.message);
    end
    
    % Handle first row (header)
    if isempty(options.skipFirstRow)
        % Auto-detect: skip first row if row count exceeds expected (first dimension)
        if size(data, 1) > options.dimensions(1)
            data = data(2:end, :);
        end
    elseif options.skipFirstRow
        % Skip first row if requested
        data = data(2:end, :);
    end
    
    % Extract features (all columns except last)
    X = data(:, 1:end-1);
    
    % Remove entries with labels 1 or 2
    labels = data(:, end);
    validIndices = ~(labels == 1 | labels == 2);  % Fixed: added == operators
    X = X(validIndices, :);
    labels = labels(validIndices);
    
    % Print matrix dimensions if verbose
    [rows, cols] = size(X);
    if options.verbose
        fprintf('Matrix size: %d × %d\n', rows, cols);
    end
    
    % Validate dimensions before reshaping
    totalElements = prod(options.dimensions);
    if numel(X) ~= totalElements
        error(['Cannot reshape matrix to specified dimensions. ', ...
            'Matrix has %d elements, but reshape requires %d elements.'], ...
            numel(X), totalElements);
    end
    
    % Reshape matrix
    fnirs_data = reshape(X, options.dimensions);
end

% Function to convert fNIRS data to BBCI toolbox format
function [cnt, mrk] = convertNirsToBBCItoolbox(fnirs_data, labels, ch_names, fs)
% Convert fNIRS data to BBCI toolbox format
%
% Inputs:
%   fnirs_data - Shape [trials × channels × timepoints]
%   labels - 120×1 array with values 4-7 indicating classes
%   ch_names - Cell array containing channel names (optional)
%   fs - Sampling rate (optional, default is 10)
%
% Outputs:
%   cnt - Structure containing data, channel names, and sampling rate
%   mrk - Structure containing timestamps and class labels

    % Set default sampling rate
    if nargin < 4 || isempty(fs)
        fs = 10;
    end
    
    % Create cnt structure
    cnt = struct();
    
    % Reshape data to [timepoints × channels] format
    cnt.x = [];
    for m = 1:size(fnirs_data, 1)  % Iterate through all trials
        event_data = squeeze(fnirs_data(m, :, :));  % Extract channels×timepoints for this trial
        cnt.x = [cnt.x; event_data'];  % Transpose to timepoints×channels and append
    end
    
    % Set channel names
    num_channels = size(fnirs_data, 2);
    if nargin < 3 || isempty(ch_names)
        % Create default channel names
        clab_cell = cell(1, num_channels);
        for m = 1:num_channels
            clab_cell{m} = sprintf('Ch%d', m);  % Create default channel name
        end
        cnt.clab = clab_cell;
    else
        cnt.clab = ch_names;
    end
    
    % Set sampling rate
    cnt.fs = double(fs);
    
    % Create mrk structure
    mrk = struct();
    
    % Set timestamps
    si = 1000/cnt.fs;  % Time interval (milliseconds)
    event_length = size(fnirs_data, 3) * si;  % Length of each trial
    mrk.time = zeros(1, size(fnirs_data, 1));  % 1×number of trials
    for m = 1:size(fnirs_data, 1)
        mrk.time(m) = (m-1) * event_length + 5000;  % Sequential timestamps, +5000 means start from 5 seconds
    end
    
    % Convert labels to one-hot encoding
    mrk.y = false(4, length(labels));  % Initialize as 4×trials logical array, all false
    for m = 1:length(labels)
        mrk.y(labels(m)-3, m) = true;  % Set corresponding class to true, 4-7 minus 3 gives 1-4
    end
    
    % Set class names
    mrk.className = {'Left to right', 'Top to bottom', 'Top left to bottom right', 'Top right to bottom left'};
end

% Bandpass filter function
function filtered_data = applyBandpassFilter(data, lowFreq, highFreq)
% Apply bandpass filter to EEG/signal data
%
% Inputs:
%   data - Structure with field .x containing signal data [timepoints × channels × trials]
%   lowFreq - Lower cutoff frequency (Hz)
%   highFreq - Upper cutoff frequency (Hz)
%
% Output:
%   filtered_data - Structure with filtered signal data

    % Copy input data structure
    filtered_data = data;
    
    % Get signal parameters
    [~, ~, nTrials] = size(data.x);
    
    % Design Butterworth bandpass filter
    fs = 250; % Sampling rate 250Hz
    filterOrder = 4;
    [b, a] = butter(filterOrder, [lowFreq, highFreq]/(fs/2), 'bandpass');
    
    % Process all channels for each trial
    for trial = 1:nTrials
        % Convert to double and process all channels at once
        filtered_data.x(:, :, trial) = filtfilt(b, a, double(data.x(:, :, trial))')';
    end
end
