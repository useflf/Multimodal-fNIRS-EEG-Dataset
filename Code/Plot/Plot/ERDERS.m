%% ERD/ERS Curves for Specific Frequency Bands - Four-Class Task (Individual Plots)

% Subject configuration
subject_index = 2; 
subject_id = sprintf('%02d', subject_index);
filename = sprintf('subject_%s_motor_imagery_events.mat', subject_id);

% Data directory configuration
time = '0927';
data_root = 'D:\Code\Dataset_4-classification MI';
data_file_path = fullfile(data_root, 'Data', 'Dataset v2', time, 'EEG', filename);

% Load data
load(data_file_path)

% Select C3 and C4 channels (motor cortex channels)
selected_channels = [17, 21];

% Prepare data with correct dimensions
x_train = zeros(length(selected_channels), size(preprocessed_data, 2), size(preprocessed_data, 3));
for i = 1:length(selected_channels)
    channel_data = preprocessed_data(selected_channels(i), :, :);
    x_train(i, :, :) = reshape(channel_data, [1, size(channel_data, 2), size(channel_data, 3)]);
end
fs = srate;

% Prepare labels
[~, y_label] = max(label, [], 1);

% Task class names
task_names = {'Left to right', 'Top to bottom', 'Top left to bottom right', 'Top right to bottom left'};

%% Alpha band (8-12Hz) ERD/ERS curves - separate plot for each task
for task = 1:4
    % Select data for current task
    task_indices = find(y_label == task);
    if isempty(task_indices)
        warning(['No data found for task ', num2str(task)]);
        continue;
    end
    
    x_task = x_train(:, :, task_indices);
    y_task = ones(size(task_indices));
    
    % Create figure
    figure('Name', ['Alpha Band ERD/ERS Curves - ', task_names{task}], 'Position', [100, 100, 1000, 500]);
    annotation('textbox', [0.1, 0.98, 0.8, 0.02], ...
        'String', ['Four-Class Task - ', task_names{task}, ' Alpha Band (8-12Hz) ERD/ERS Curves'], ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'FitBoxToText', false);
    
    % Plot ERD/ERS curves
    plotERDERS(x_task, y_task, 8, 12, fs, 5, 15, ch_names, selected_channels, [-100, 250], [-5, 10]);
end

%% Beta band (13-30Hz) ERD/ERS curves - separate plot for each task
for task = 1:4
    % Select data for current task
    task_indices = find(y_label == task);
    if isempty(task_indices)
        warning(['No data found for task ', num2str(task)]);
        continue;
    end
    
    x_task = x_train(:, :, task_indices);
    y_task = ones(size(task_indices));
    
    % Create figure
    figure('Name', ['Beta band ERD/ERS Curves - ', task_names{task}], 'Position', [100, 100, 1000, 500]);
    annotation('textbox', [0.1, 0.98, 0.8, 0.02], ...
        'String', ['Four-Class Task - ', task_names{task}, ' Beta Band (13-30Hz) ERD/ERS Curves'], ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        'FontSize', 14, 'FontWeight', 'bold', 'EdgeColor', 'none', 'FitBoxToText', false);
    
    % Plot ERD/ERS curves
    plotERDERS(x_task, y_task, 13, 30, fs, 5, 15, ch_names, selected_channels, [-100, 100], [-5, 10]);
end

%% Helper Functions

function plotERDERS(data, labels, flow, fhigh, fs, baseline_end, trial_length, ch_names, selected_channels, ylim_range, time_range)
    % Plot ERD/ERS curves
    %
    % Inputs:
    %   data            - Input data [channels x time_points x trials]
    %   labels          - Task labels [1 x trials]
    %   flow, fhigh     - Lower and upper frequency band limits (Hz)
    %   fs              - Sampling rate (Hz)
    %   baseline_end    - Baseline period end time (seconds)
    %   trial_length    - Total trial length (seconds)
    %   ch_names        - Channel names cell array
    %   selected_channels - Selected channel indices
    %   ylim_range      - Y-axis range [lower upper], e.g., [-100 250]
    %   time_range      - Time axis range [start end], e.g., [-2 10]
    
    % Set default time range if not provided
    if nargin < 11 || isempty(time_range)
        time_range = [0, trial_length];
    end
    
    % Plot style configuration
    colors = {'r', 'b'};
    line_width = 2;
    smoothPara = 80; % Smoothing window size
    
    % Calculate and plot ERD/ERS curves
    hold on;
    for ch = 1:size(data, 1)
        % Get all trials for current channel
        channel_trials = squeeze(data(ch, :, :));
        
        % Compute ERD/ERS
        [erders, t] = computeERDERS(channel_trials, flow, fhigh, fs, ...
            round(baseline_end*fs), round(trial_length*fs), smoothPara);
        
        % Apply smoothing
        erders = movingAverage(erders, smoothPara);
        
        % Adjust time axis (shift left by baseline_end seconds)
        t_adjusted = t - baseline_end;
        
        % Plot curve
        plot(t_adjusted, erders, colors{ch}, 'LineWidth', line_width);
    end
    
    % Set axis limits and labels
    xlim(time_range);
    xlabel('Time (s)');
    ylabel('Relative Power Change (%)');
    
    % Set Y-axis range
    if nargin >= 10 && ~isempty(ylim_range)
        ylim(ylim_range);
    else
        ylim([-100 100]);
    end
    grid on;
    
    % Add reference lines
    plot([baseline_end baseline_end], ylim_range, 'k:'); % Baseline end
    plot([0 0], ylim_range, 'k:'); % Zero time point
    plot([time_range(1) time_range(2)], [0 0], 'k--'); % Zero percent reference
    
    % Add legend
    legend_labels = cell(size(data, 1), 1);
    for i = 1:size(data, 1)
        legend_labels{i} = ['Channel ', ch_names{selected_channels(i)}];
    end
    legend(legend_labels, 'Location', 'best');
    hold off;
end

function [erders, t] = computeERDERS(trials, flow, fhigh, fs, baseline_samples, total_samples, smoothPara)
    % Compute ERD/ERS (Event-Related Desynchronization/Synchronization)
    %
    % Inputs:
    %   trials          - Input trials [time_points x trials]
    %   flow, fhigh     - Lower and upper frequency band limits (Hz)
    %   fs              - Sampling rate (Hz)
    %   baseline_samples - Number of baseline samples
    %   total_samples   - Total number of samples
    %   smoothPara      - Smoothing window size
    %
    % Outputs:
    %   erders          - ERD/ERS values (%)
    %   t               - Time vector (seconds)
    
    % Ensure total samples doesn't exceed available data
    total_samples = min(total_samples, size(trials, 1));
    
    % Initialize
    n_trials = size(trials, 2);
    filtered_data = zeros(size(trials));
    butterOrder = 6; % Butterworth filter order
    
    % Filter each trial separately
    for i = 1:n_trials
        filtered_data(:, i) = filter_param(trials(:, i), flow, fhigh, fs, butterOrder);
    end
    
    % Calculate power
    power_data = filtered_data.^2;
    
    % Average power across all trials
    avg_power = mean(power_data, 2);
    avg_power = avg_power(1:total_samples);
    
    % Apply smoothing
    avg_power = movingAverage(avg_power, smoothPara);
    
    % Calculate baseline power (average during baseline period)
    baseline_power = mean(avg_power(1:baseline_samples));
    
    % Calculate ERD/ERS
    erders = ((avg_power - baseline_power) / baseline_power) * 100;
    
    % Create time vector
    t = (0:length(erders)-1) / fs;
end

function smoothed = movingAverage(data, window_size)
    % Custom moving average smoothing function
    %
    % Inputs:
    %   data        - Input data vector
    %   window_size - Window size (must be odd)
    %
    % Output:
    %   smoothed    - Smoothed data vector
    
    % Ensure window size is odd
    if mod(window_size, 2) == 0
        window_size = window_size + 1;
    end
    
    n = length(data);
    half_window = floor(window_size / 2);
    smoothed = zeros(size(data));
    
    % Pad data to handle boundaries
    padded_data = [repmat(data(1), [half_window, 1]); data; repmat(data(end), [half_window, 1])];
    
    % Apply moving average
    for i = 1:n
        smoothed(i) = mean(padded_data(i:i+window_size-1));
    end
end

function filterdata = filter_param(data, low, high, sampleRate, filterorder)
    % Bandpass filter for EEG data
    %
    % Inputs:
    %   data        - EEG data to be filtered
    %   low         - High-pass filter cutoff frequency (Hz)
    %   high        - Low-pass filter cutoff frequency (Hz)
    %   sampleRate  - Sampling rate (Hz)
    %   filterorder - Butterworth filter order
    %
    % Output:
    %   filterdata  - Filtered EEG data
    
    % Calculate normalized cutoff frequencies
    filtercutoff = [low*2/sampleRate high*2/sampleRate];
    
    % Design Butterworth filter
    [filterParamB, filterParamA] = butter(filterorder, filtercutoff);
    
    % Apply filter
    filterdata = filter(filterParamB, filterParamA, data);
end
