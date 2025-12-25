%% EEG Motor Imagery Classification using Filter Bank CSP and LDA

% Load configuration file
data_root = 'D:\Code\Dataset_4-classification MI';
info_file_path = fullfile(data_root, 'info.json');

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
            
            % Data file path
            data_file_path = fullfile(data_root, 'Data', 'Dataset v2', time, 'EEG', filename);
            
            % Load data: preprocessed_data (64×6250×120), label (4×120), ch_names, srate (250Hz)
            load(data_file_path)
            disp(['Data directory:', data_file_path])
            
            whos preprocessed_data label ch_names srate
            
            %% Prepare cnt structure for BBCI toolbox
            cnt = struct();
            
            % Create data matrix [time_points × channels]
            cnt.x = [];
            for m = 1:size(preprocessed_data, 3)
                event_data = preprocessed_data(:, :, m);
                cnt.x = [cnt.x; double(event_data')];
            end
            
            % Ensure channel labels are cell array
            if ~iscell(ch_names)
                clab_cell = cell(1, length(ch_names));
                for n = 1:length(ch_names)
                    if ischar(ch_names(n)) || isstring(ch_names(n))
                        clab_cell{n} = char(ch_names(n));
                    else
                        clab_cell{n} = sprintf('Ch%d', n);
                    end
                end
                cnt.clab = clab_cell;
            else
                cnt.clab = ch_names;
            end
            
            cnt.fs = double(srate);
            
            %% Prepare mrk structure
            mrk = struct();
            
            % Set time markers (row vector)
            si = 1000/cnt.fs;
            event_length = 6250 * si;
            mrk.time = zeros(1, 120);
            for m = 1:120
                mrk.time(m) = 1250 * si + (m-1) * event_length;
            end
            
            mrk.y = logical(label);
            mrk.className = {'Left to right', 'Top to bottom', 'Top left to bottom right', 'Top right to bottom left'};
            
            %% Initialize BBCI toolbox
            EegMyDataDir = fullfile('D:','Code','Dataset_4-classification MI','Data', 'Dataset v2', '0927', 'EEG');
            TemDir = fullfile('D:','Code','Dataset_4-classification MI','Data', 'Dataset v2', '0927', 'EEG', 'tmp');
            WorkingDir = fullfile('D:','Code','Dataset_4-classification MI','CSP-LDA');
            startup_bbci_toolbox('DataDir',EegMyDataDir,'TmpDir',TemDir);
            BTB.History = 0;
            
            % Segment data
            ival_epo = [-5000 20000];
            epo = proc_segmentation(cnt, mrk, ival_epo);
            
            %% Channel selection
            % Motor cortex channels
            MotorChannel = {'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', ...
                'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', ...
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'};
            
            % Parietal channels
            ParietalChannel = {'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'};
            
            cnt_org = cnt;
            cnt = proc_selectChannels(cnt, [MotorChannel, ParietalChannel]);
            
            %% CSP frequency band selection
            band_csp = select_bandnarrow(cnt, mrk, [0 10]*1000);
            
            % Design Chebyshev Type II bandpass filter
            Wp = band_csp/epo.fs*2;
            Ws = [band_csp(1)-3, band_csp(end)+3]/epo.fs*2;
            Rp = 3; % Passband ripple (dB)
            Rs = 30; % Stopband attenuation (dB)
            [ord, Ws] = cheb2ord(Wp, Ws, Rp, Rs);
            [filt_b, filt_a] = cheby2(ord, Rs, Ws);
            epo = proc_filtfilt(epo, filt_b, filt_a);
            
            %% Moving time window classification
            StepSize = 1*1000; % ms
            WindowSize = 3*1000; % ms
            ival_start = (ival_epo(1):StepSize:ival_epo(end)-WindowSize)';
            ival_end = ival_start + WindowSize;
            ival = [ival_start, ival_end];
            nStep = length(ival);
            
            % Define filter bank frequency bands (Hz)
            nBands = 9;
            freqBands = zeros(nBands, 2);
            for m = 1:nBands
                freqBands(m,:) = [4+(m-1)*4, 8+(m-1)*4];
            end
            
            % Select time window segments
            for stepIdx = 1:nStep
                segment{stepIdx} = proc_selectIval(epo, ival(stepIdx,:));
            end
            
            %% Cross-validation setup
            nShift = 2; % Number of repetitions
            nFold = 5; % Number of folds
            group = epo.y;
            nClasses = size(group, 1); % 4 classes
            nFilters = 2; % CSP filters per class per band
            
            %% Motor imagery classification with FBCSP
            for shiftIdx = 1:nShift
                indices{shiftIdx} = crossvalind('Kfold', full(vec2ind(group)), nFold);
                
                for stepIdx = 1:nStep
                    fprintf('Motor Imagery, Repeat:%d/%d, Step:%d/%d\n', shiftIdx, nShift, stepIdx, nStep);
                    
                    % Pre-compute filtered data for all bands
                    filtered_data_all_bands = cell(nBands, 1);
                    eeg_all_data = segment{stepIdx};
                    
                    for bandIdx = 1:nBands
                        filtered_data_all_bands{bandIdx} = applyBandpassFilter(eeg_all_data, freqBands(bandIdx,1), freqBands(bandIdx,2));
                    end
                    
                    for foldIdx = 1:nFold
                        test = (indices{shiftIdx} == foldIdx);
                        train = ~test;
                        
                        all_data = segment{stepIdx};
                        
                        % One-vs-rest classification for each class
                        fv_test.eeg = cell(nClasses, 1);
                        C.eeg = cell(nClasses, 1);
                        
                        for classIdx = 1:nClasses
                            % Create binary labels for current class vs others
                            curr_class_idx = find(group(classIdx, train) == 1);
                            other_class_idx = find(group(classIdx, train) ~= 1);
                            
                            binary_labels = zeros(2, length(train));
                            binary_labels(1, curr_class_idx) = 1;
                            binary_labels(2, other_class_idx) = 1;
                            
                            % FBCSP: Process all frequency bands
                            fv_train_allbands = [];
                            fv_test_allbands = [];
                            
                            for bandIdx = 1:nBands
                                all_data_band = filtered_data_all_bands{bandIdx};
                                
                                % Split train and test data
                                x_train_band = struct();
                                x_train_band.x = all_data_band.x(:,:,train);
                                x_train_band.y = all_data_band.y(:,train);
                                x_train_band.clab = all_data_band.clab;
                                
                                x_test_band = struct();
                                x_test_band.x = all_data_band.x(:,:,test);
                                x_test_band.y = all_data_band.y(:,test);
                                x_test_band.clab = all_data_band.clab;
                                
                                % Set binary labels
                                x_train_binary = x_train_band;
                                third_dimension_size = size(x_train_binary.x, 3);
                                x_train_binary.y = binary_labels(:,1:third_dimension_size);
                                
                                % Apply CSP
                                [csp_train, CSP_W, CSP_EIG, CSP_A] = proc_cspAuto(x_train_binary);
                                csp_train.x = csp_train.x(:,[1 2 end-1 end],:);
                                
                                % Apply same CSP to test data
                                csp_test = struct();
                                for testIdx = 1:size(find(test==1),1)
                                    csp_test.x(:,:,testIdx) = x_test_band.x(:,:,testIdx)*CSP_W;
                                end
                                csp_test.x = csp_test.x(:,[1 2 end-1 end],:);
                                csp_test.y = x_test_band.y;
                                csp_test.clab = x_test_band.clab;
                                
                                % Extract features
                                var_train = proc_variance(csp_train);
                                logvar_train = proc_logarithm(var_train);
                                var_test = proc_variance(csp_test);
                                logvar_test = proc_logarithm(var_test);
                                
                                % Concatenate features from all bands
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
                            
                            fv_test.eeg{classIdx} = fv_test_allbands;
                            
                            % Train RLDA classifier
                            C.eeg{classIdx} = train_RLDAshrink(fv_train_allbands, binary_labels);
                        end
                        
                        % Predict using all classifiers (select class with minimum decision value)
                        y_test = vec2ind(group(:,test));
                        num_test_samples = length(y_test);
                        decision_values = zeros(nClasses, num_test_samples);
                        
                        for classIdx = 1:nClasses
                            [~, decision_val] = applyRLDAClassifier(fv_test.eeg{classIdx}, C.eeg{classIdx});
                            decision_values(classIdx, :) = decision_val;
                        end
                        
                        [~, predicted_classes] = min(decision_values, [], 1);
                        grouphat.eeg(foldIdx,:) = predicted_classes;
                        
                        % Compute confusion matrix
                        cmat.eeg(:,:,foldIdx) = confusionmat(y_test, grouphat.eeg(foldIdx,:));
                    end
                    
                    % Calculate average accuracy
                    acc.eeg(shiftIdx, stepIdx) = trace(sum(cmat.eeg,3)) / sum(sum(sum(cmat.eeg,3)));
                end
            end
            
            mean_acc.eeg = mean(acc.eeg, 1)';
            
            %% Plot results
            T = (ival(:,2)/1000)';
            fig = figure();
            plot(T, mean_acc.eeg, 'b');
            legend('Motor Imagery (FBCSP)');
            xlim([T(1) T(end-6)]);
            ylim([0.2 0.6]);
            grid on;
            
            % Save figure
            filename = sprintf('%s_subject_%s_EEG.fig', time, subject_id);
            disp(['Filename:', filename]);
            saveas(fig, filename);
            close(fig);
        end
    end
end

%% Helper Functions

function filtered_data = applyBandpassFilter(data, lowFreq, highFreq)
    % Apply Butterworth bandpass filter to EEG data
    %
    % Inputs:
    %   data     - EEG data structure with field .x [channels × time_points × trials]
    %   lowFreq  - Lower cutoff frequency (Hz)
    %   highFreq - Upper cutoff frequency (Hz)
    %
    % Output:
    %   filtered_data - Filtered EEG data structure
    
    filtered_data = data;
    [~, ~, nTrials] = size(data.x);
    
    % Design Butterworth bandpass filter
    fs = 250; % Sampling rate 250Hz
    filterOrder = 4;
    [b, a] = butter(filterOrder, [lowFreq, highFreq]/(fs/2), 'bandpass');
    
    % Filter all channels for each trial
    for trial = 1:nTrials
        filtered_data.x(:, :, trial) = filtfilt(b, a, double(data.x(:, :, trial))')';
    end
end