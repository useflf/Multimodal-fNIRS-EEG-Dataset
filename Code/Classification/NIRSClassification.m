%% fNIRS Motor Imagery Classification using RLDA

% Load configuration file
data_root = 'D:\Code\Dataset_4-classification MI';
info_file_path = fullfile(data_root, 'info.json');

% Working directories
EegMyDataDir = 'D:\Code\Dataset_4-classification MI\Data\Dataset v2';
TemDir = 'D:\Code\Dataset_4-classification MI\Data\Dataset v2';
WorkingDir = 'D:\Code\Dataset_4-classification MI\CSP-LDA';

% Read data information and channel-ID mapping from JSON file
try
    info_content = fileread(info_file_path);
    info = jsondecode(info_content);
catch
    fprintf('Unable to load configuration file\n');
    return;
end

% Iterate through all time fields in data structure
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
            
            % Subject configuration
            subject_index = k;
            subject_id = sprintf('%02d', subject_index);
            filename = sprintf('subject_%s_motor_imagery_events.mat', subject_id);
            nirs_folder_name = sprintf('sub_%s', subject_id);
            
            % Data file paths
            hbo_filepath = fullfile(data_root, 'Data', 'Dataset v2', time, 'NIRS', nirs_folder_name, 'hbo.csv');
            hbr_filepath = fullfile(data_root, 'Data', 'Dataset v2', time, 'NIRS', nirs_folder_name, 'hbr.csv');
            
            if ~exist(hbo_filepath, 'file') || ~exist(hbr_filepath, 'file')
                fprintf('File not found, skipping: %s\n', nirs_folder_name);
                continue;
            end
            
            % Load data options
            opts = struct('skipFirstRow', true, 'dimensions', [120, 51, 301], 'verbose', true);
            
            % Load HbO and HbR data
            [hbo_fnirs_data, labels] = loadFnirsData(hbo_filepath, opts);
            [hbr_fnirs_data, ~] = loadFnirsData(hbr_filepath, opts);
            
            % Convert to BBCI toolbox format
            [hbo_cnt, mrk] = convertNirsToBBCItoolbox(hbo_fnirs_data, labels);
            [hbr_cnt, ~] = convertNirsToBBCItoolbox(hbr_fnirs_data, labels);
            
            %% Initialize BBCI toolbox
            NirsMyDataDir = fullfile('D:', 'Code', 'Dataset_4-classification MI', 'Data', 'fnirs_processed');
            TemDir = fullfile('D:', 'Code', 'Dataset_4-classification MI', 'Data', 'fnirs_processed');
            WorkingDir = fullfile('D:', 'Code', 'Dataset_4-classification MI', 'CSP-LDA');
            startup_bbci_toolbox('DataDir', NirsMyDataDir, 'TmpDir', TemDir);
            BTB.History = 0;
            
            %% fNIRS channel selection by brain region
            FrontalChannel = {'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', ...
                'Ch35', 'Ch36', 'Ch37', 'Ch38', 'Ch39', 'Ch40', 'Ch41', ...
                'Ch18', 'Ch19', 'Ch20', 'Ch21', 'Ch22', 'Ch23', 'Ch24'};
            
            MotorChannel = {'Ch9', 'Ch10', 'Ch12', ...
                'Ch42', 'Ch43', 'Ch44', 'Ch45', 'Ch46', ...
                'Ch25', 'Ch26', 'Ch28'};
            
            ParietalChannel = {'Ch14', 'Ch15', 'Ch17', ...
                'Ch47', 'Ch48', 'Ch49', 'Ch50', 'Ch51', ...
                'Ch30', 'Ch31', 'Ch33', 'Ch34'};
            
            TemporalChannel = {'Ch8', 'Ch11', 'Ch13', 'Ch16', ...
                'Ch27', 'Ch29', 'Ch32'};
            
            % Select channels
            hbo_cnt = proc_selectChannels(hbo_cnt, [FrontalChannel, MotorChannel, ParietalChannel]);
            hbr_cnt = proc_selectChannels(hbr_cnt, [FrontalChannel, MotorChannel, ParietalChannel]);
            
            % Segment data
            ival_epo = [-5000 20000];
            epo.oxy = proc_segmentation(hbo_cnt, mrk, ival_epo);
            epo.deoxy = proc_segmentation(hbr_cnt, mrk, ival_epo);
            
            %% Moving time window classification
            nShift = 5;
            nFold = 5;
            StepSize = 1*1000; % ms
            WindowSize = 3*1000; % ms
            ival_start = (ival_epo(1):StepSize:ival_epo(end)-WindowSize)';
            ival_end = ival_start + WindowSize;
            ival = [ival_start, ival_end];
            nStep = length(ival);
            
            % Calculate mean features
            for stepIdx = 1:nStep
                ave.deoxy{stepIdx} = proc_meanAcrossTime(epo.deoxy, ival(stepIdx,:));
                ave.oxy{stepIdx} = proc_meanAcrossTime(epo.oxy, ival(stepIdx,:));
            end
            
            % Calculate slope features
            for stepIdx = 1:nStep
                slope.deoxy{stepIdx} = proc_slopeAcrossTime(epo.deoxy, ival(stepIdx,:));
                slope.oxy{stepIdx} = proc_slopeAcrossTime(epo.oxy, ival(stepIdx,:));
            end
            
            %% Cross-validation setup
            group = epo.deoxy.y;
            nClasses = size(group, 1); % 4 classes
            
            %% Motor imagery classification
            for shiftIdx = 1:nShift
                indices{shiftIdx} = crossvalind('Kfold', length(vec2ind(group)), nFold);
                
                for stepIdx = 1:nStep
                    fprintf('Motor Imagery, Repeat:%d/%d, Step:%d/%d\n', shiftIdx, nShift, stepIdx, nStep);
                    
                    for foldIdx = 1:nFold
                        test = (indices{shiftIdx} == foldIdx);
                        train = ~test;
                        
                        % Initialize classifiers
                        C.deoxy = cell(nClasses, 1);
                        C.oxy = cell(nClasses, 1);
                        
                        % Prepare training and test data
                        x_train.deoxy.x = [squeeze(ave.deoxy{stepIdx}.x(:,:,train)); squeeze(slope.deoxy{stepIdx}.x(:,:,train))];
                        x_train.deoxy.y = squeeze(group(:,train));
                        x_train.deoxy.clab = ave.deoxy{stepIdx}.clab;
                        
                        x_train.oxy.x = [squeeze(ave.oxy{stepIdx}.x(:,:,train)); squeeze(slope.oxy{stepIdx}.x(:,:,train))];
                        x_train.oxy.y = squeeze(group(:,train));
                        x_train.oxy.clab = ave.oxy{stepIdx}.clab;
                        
                        x_test.deoxy.x = [squeeze(ave.deoxy{stepIdx}.x(:,:,test)); squeeze(slope.deoxy{stepIdx}.x(:,:,test))];
                        x_test.deoxy.y = squeeze(group(:,test));
                        x_test.deoxy.clab = ave.deoxy{stepIdx}.clab;
                        
                        x_test.oxy.x = [squeeze(ave.oxy{stepIdx}.x(:,:,test)); squeeze(slope.oxy{stepIdx}.x(:,:,test))];
                        x_test.oxy.y = squeeze(group(:,test));
                        x_test.oxy.clab = ave.oxy{stepIdx}.clab;
                        
                        % Train one-vs-rest classifiers for each class
                        for classIdx = 1:nClasses
                            % Create binary labels
                            curr_class_idx = find(group(classIdx,train) == 1);
                            other_class_idx = find(group(classIdx,train) ~= 1);
                            
                            binary_labels = zeros(2, length(curr_class_idx)+length(other_class_idx));
                            binary_labels(1, curr_class_idx) = 1;
                            binary_labels(2, other_class_idx) = 1;
                            
                            % Train deoxy classifier
                            fv_train_deoxy = struct();
                            fv_train_deoxy.x = x_train.deoxy.x;
                            second_dimension_size = size(fv_train_deoxy.x, 2);
                            fv_train_deoxy.y = binary_labels(:,1:second_dimension_size);
                            C.deoxy{classIdx} = train_RLDAshrink(fv_train_deoxy.x, fv_train_deoxy.y);
                            
                            % Train oxy classifier
                            fv_train_oxy = struct();
                            fv_train_oxy.x = x_train.oxy.x;
                            fv_train_oxy.y = binary_labels(:,1:second_dimension_size);
                            C.oxy{classIdx} = train_RLDAshrink(fv_train_oxy.x, fv_train_oxy.y);
                        end
                        
                        % Predict using all classifiers
                        y_test = vec2ind(x_test.deoxy.y);
                        num_test_samples = length(y_test);
                        
                        decision_values_deoxy = zeros(nClasses, num_test_samples);
                        decision_values_oxy = zeros(nClasses, num_test_samples);
                        
                        for classIdx = 1:nClasses
                            fv_test_deoxy = struct();
                            fv_test_deoxy.x = x_test.deoxy.x;
                            fv_test_oxy = struct();
                            fv_test_oxy.x = x_test.oxy.x;
                            
                            [~, decision_val_deoxy] = applyRLDAClassifier(fv_test_deoxy, C.deoxy{classIdx});
                            decision_values_deoxy(classIdx, :) = decision_val_deoxy;
                            
                            [~, decision_val_oxy] = applyRLDAClassifier(fv_test_oxy, C.oxy{classIdx});
                            decision_values_oxy(classIdx, :) = decision_val_oxy;
                        end
                        
                        % Select class with minimum decision value
                        [~, predicted_classes_deoxy] = min(decision_values_deoxy, [], 1);
                        [~, predicted_classes_oxy] = min(decision_values_oxy, [], 1);
                        
                        grouphat.deoxy(foldIdx,:) = predicted_classes_deoxy;
                        grouphat.oxy(foldIdx,:) = predicted_classes_oxy;
                        
                        % Compute confusion matrices
                        cmat.deoxy(:,:,foldIdx) = confusionmat(y_test, grouphat.deoxy(foldIdx,:));
                        cmat.oxy(:,:,foldIdx) = confusionmat(y_test, grouphat.oxy(foldIdx,:));
                    end
                    
                    % Calculate accuracy
                    acc.deoxy(shiftIdx,stepIdx) = trace(sum(cmat.deoxy,3)) / sum(sum(sum(cmat.deoxy,3)));
                    acc.oxy(shiftIdx,stepIdx) = trace(sum(cmat.oxy,3)) / sum(sum(sum(cmat.oxy,3)));
                end
            end
            
            mean_acc.deoxy = mean(acc.deoxy,1)';
            mean_acc.oxy = mean(acc.oxy,1)';
            result = [mean_acc.deoxy, mean_acc.oxy];
            
            %% Plot results
            T = (ival(:,2)/1000)';
            fig = figure();
            plot(T, result);
            legend('deoxy', 'oxy');
            xlim([T(1) T(end)]);
            ylim([0 0.5]);
            grid on;
            
            % Save figure
            filename = sprintf('%s_subject_%s_NIRS.fig', time, subject_id);
            disp(['Filename:', filename]);
            saveas(fig, filename);
            close(fig);
        end
    end
end

%% Helper Functions

function [fnirs_data, labels] = loadFnirsData(filepath, options)
    % Load and process fNIRS data
    %
    % Inputs:
    %   filepath - CSV file path
    %   options  - Structure with optional fields:
    %       .skipFirstRow - Whether to skip first row (default: auto-detect)
    %       .dimensions   - 3-element vector for reshape (default: [120, 51, 301])
    %       .verbose      - Whether to print info (default: true)
    %
    % Outputs:
    %   fnirs_data - Reshaped fNIRS data
    %   labels     - Task labels
    
    if ~isfile(filepath)
        error('File not found: %s', filepath);
    end
    
    % Default options
    defaultOpts = struct('skipFirstRow', [], 'dimensions', [120, 51, 301], 'verbose', true);
    
    if nargin < 2
        options = defaultOpts;
    else
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
        error('Error: %s\n%s', filepath, e.message);
    end
    
    % Handle header row
    if isempty(options.skipFirstRow)
        if size(data, 1) > options.dimensions(1)
            data = data(2:end, :);
        end
    elseif options.skipFirstRow
        data = data(2:end, :);
    end
    
    % Extract features and labels
    X = data(:, 1:end-1);
    labels = data(:, end);
    
    % Remove entries with labels 1 or 2
    validIndices = ~(labels == 1 | labels == 2);
    X = X(validIndices, :);
    labels = labels(validIndices);
    
    [rows, cols] = size(X);
    if options.verbose
        fprintf('Matrix size: %d × %d\n', rows, cols);
    end
    
    % Validate dimensions before reshaping
    totalElements = prod(options.dimensions);
    if numel(X) ~= totalElements
        error('Cannot reshape matrix to specified dimensions. Matrix has %d elements, but reshape requires %d elements.', numel(X), totalElements);
    end
    
    fnirs_data = reshape(X, options.dimensions);
end

function [cnt, mrk] = convertNirsToBBCItoolbox(fnirs_data, labels, ch_names, fs)
    % Convert fNIRS data to BBCI toolbox format
    %
    % Inputs:
    %   fnirs_data - Data array [trials × channels × time_points]
    %   labels     - Label array [120×1] with values 4-7
    %   ch_names   - Channel names cell array (optional)
    %   fs         - Sampling rate (optional, default: 10)
    %
    % Outputs:
    %   cnt - Structure with data, channel names, and sampling rate
    %   mrk - Structure with timestamps and class labels
    
    if nargin < 4 || isempty(fs)
        fs = 10;
    end
    
    % Create cnt structure
    cnt = struct();
    cnt.x = [];
    for m = 1:size(fnirs_data, 1)
        event_data = squeeze(fnirs_data(m, :, :));
        cnt.x = [cnt.x; event_data'];
    end
    
    % Set channel names
    num_channels = size(fnirs_data, 2);
    if nargin < 3 || isempty(ch_names)
        clab_cell = cell(1, num_channels);
        for m = 1:num_channels
            clab_cell{m} = sprintf('Ch%d', m);
        end
        cnt.clab = clab_cell;
    else
        cnt.clab = ch_names;
    end
    
    cnt.fs = double(fs);
    
    % Create mrk structure
    mrk = struct();
    si = 1000/cnt.fs;
    event_length = size(fnirs_data, 3) * si;
    mrk.time = zeros(1, size(fnirs_data, 1));
    
    for m = 1:size(fnirs_data, 1)
        mrk.time(m) = (m-1) * event_length + 5000;
    end
    
    % Convert labels to one-hot encoding
    mrk.y = false(4, length(labels));
    for m = 1:length(labels)
        mrk.y(labels(m)-3, m) = true;
    end
    
    mrk.className = {'Left to right', 'Top to bottom', 'Top left to bottom right', 'Top right to bottom left'};
end