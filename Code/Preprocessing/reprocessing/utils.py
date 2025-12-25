import mne
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
import json

def load_data_from_json(file_path):
    """Load data from JSON file"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_raw_eeg_data(file_paths):
    """Load raw EEG data files
    
    Parameters
    ----------
    file_paths : list
        List of EEG data file paths
        
    Returns
    -------
    mne.io.Raw
        Merged raw EEG data
    """
    raws = []
    for file_path in file_paths:
        try:
            raw_data = mne.io.read_raw_curry(file_path, preload=True)
            raws.append(raw_data)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
    
    if not raws:
        raise ValueError("No valid EEG data files")
    
    # Merge all data
    data = mne.concatenate_raws(raws)
    return data

def display_data_info(raw_data):
    """Display basic information of EEG data
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw EEG data
    """
    print("\nInfo attributes:")
    print(raw_data.info)
    print("\nChannel names:")
    print(raw_data.info['ch_names'])
    print("\nSampling frequency:")
    print(raw_data.info['sfreq'])


def save_electrode_montage(raw_data, output_file):
    """Extract and save electrode position information
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw EEG data
    output_file : str
        Output file path
    """
    ch_names = raw_data.info['ch_names']
    chs = raw_data.info['chs']
    
    # Create a dictionary to store channel names and coordinates
    mnt = {
        'clab': np.array(ch_names[:-5], dtype=object),
        'pos_3d': np.array([ch['loc'][:3] for ch in chs][:-5]).T,
        'fs': raw_data.info['sfreq']
    }
    
    # Get directory path
    folder_path = os.path.dirname(output_file)
    # Check if directory exists, create if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save as .mat file
    savemat(output_file, mnt)
    print(f"Electrode position information saved to: {output_file}")


def apply_filters(raw_data):
    """Apply bandpass and notch filters
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw EEG data
        
    Returns
    -------
    mne.io.Raw
        Filtered EEG data
    """
    
    raw_filtered = raw_data.copy()
    # Bandpass filter: 0.5-50Hz
    raw_filtered = raw_filtered.filter(l_freq=0.5, h_freq=50)
    # 50Hz notch filter
    raw_filtered = raw_filtered.notch_filter(freqs=50, notch_widths=2)
    
    print("\n" + "="*80)
    print("Filtering complete (bandpass: 0.5-50Hz, notch: 50Hz)")
    print("="*80 + "\n")
    
    return raw_filtered


def apply_referencing(raw_data, bad_channels):
    """Apply average referencing
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        EEG data
    bad_channels : list
        Bad channels list
        
    Returns
    -------
    mne.io.Raw
        Re-referenced EEG data
    """
    # Set bad channels
    non_eeg_channels = ['M1', 'M2', 'HEO', 'VEO', 'EKG', 'EMG', 'Trigger']
    raw_data.info['bads'] = non_eeg_channels + bad_channels
    
    # Drop non-EEG channels
    raw_data.drop_channels(['HEO', 'VEO', 'EKG', 'EMG', 'Trigger', 'M1', 'M2'])
    
    # Set average reference
    raw_data.set_eeg_reference(ref_channels='average', projection=False)
    
    print("\n" + "="*80)
    print("Average referencing complete")
    print("="*80 + "\n")
    
    return raw_data

def interpolate_bad_channels(raw, bad_channels):
    """Set bad channel data to 0 then perform spherical interpolation
    
    Parameters
    ----------
    raw : mne.io.Raw
        EEG data object
    bad_channels : list of str
        Bad channel names list, e.g., ['Fp1', 'F3']
    
    Returns
    -------
    raw_interp : mne.io.Raw
        Interpolated EEG data object
    """
    if not bad_channels:
        print('No bad channels to interpolate')
        return raw.copy()
    
    # Copy data
    raw_interp = raw.copy()
    
    # Get bad channel indices
    bad_indices = [raw_interp.ch_names.index(ch) for ch in bad_channels 
                   if ch in raw_interp.ch_names]
    
    if not bad_indices:
        print('Warning: Provided bad channel names not found in data')
        return raw_interp
    
    # Set bad channel data to 0
    print(f'Setting bad channel data to 0, bad channels: {bad_channels}')
    print(f'Bad channel indices: {bad_indices}')
    
    data = raw_interp.get_data()
    for idx, ch_name in zip(bad_indices, bad_channels):
        if 0 <= idx < len(raw_interp.ch_names):
            data[idx, :] = 0
            print(f'Set channel {idx} ({ch_name}) data to 0')
        else:
            print(f'Warning: Bad channel index out of range: {idx}')
    
    # Put modified data back into raw object
    raw_interp._data = data
    
    # Mark bad channels and interpolate
    raw_interp.info['bads'] = bad_channels
    print(f'Performing spherical interpolation on bad channels: {bad_channels}')
    raw_interp.interpolate_bads(reset_bads=True, mode='accurate')
    
    return raw_interp

def create_epochs(raw_data, epoch_type='motor_imagery'):
    """Create epochs and apply baseline correction
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Preprocessed EEG data
    epoch_type : str
        Epoch type, 'motor_imagery' or 'resting_state'
        
    Returns
    -------
    mne.Epochs
        Created epochs object
    """
    # Get events
    events, event_dict = mne.events_from_annotations(raw_data, verbose=False)
    
    # Create event ID to description mapping dictionary
    id_to_desc = {v: k for k, v in event_dict.items()}
    
    # Set different parameters based on type
    if epoch_type == 'motor_imagery':
        tmin, tmax = -5, 20
        baseline = (-2, 0)
        # Filter motor imagery events
        motor_events = []
        for event in events:
            desc = id_to_desc[event[2]]
            if desc in ['4', '5', '6', '7']:
                motor_events.append(event)
        events = np.array(motor_events)
    else:  # resting_state
        tmin, tmax = 0, 60
        baseline = None
        # Filter resting state events
        resting_events = []
        for event in events:
            desc = id_to_desc[event[2]]
            if desc in ['1', '2']:
                resting_events.append(event)
        events = np.array(resting_events)
    
    # Print actual event IDs in data
    unique_events = np.unique(events[:, 2])
    print("Actual event IDs in data:", unique_events)
    print("Current event mapping:", event_dict)

    # Check for mismatched event IDs
    missing_events = set(event_dict.values()) - set(unique_events)
    if missing_events:
        print("Warning: The following event IDs do not exist in data:", missing_events)
    
    # Only use event IDs that actually exist in data
    valid_event_dict = {k: v for k, v in event_dict.items() if v in unique_events}
    
    # Create epochs
    epochs = mne.Epochs(
        raw_data,
        events,
        event_id=valid_event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=True,
    )
    
    print("\n" + "="*80)
    print(f"Epoch processing complete (type: {epoch_type})")
    if baseline:
        print(f"Time range: {tmin}s to {tmax}s, baseline: {baseline[0]}s to {baseline[1]}s")
    else:
        print(f"Time range: {tmin}s to {tmax}s, no baseline correction")
    print("="*80 + "\n")
    
    return epochs


def resample_epochs(epochs, sfreq=250):
    """Resample epoch data
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    sfreq : float
        Target sampling frequency
        
    Returns
    -------
    mne.Epochs
        Resampled epochs object
    """
    epochs_resampled = epochs.copy().resample(sfreq=sfreq)
    
    print("\n" + "="*80)
    print(f"Resampling to {sfreq}Hz complete")
    print("="*80 + "\n")
    
    return epochs_resampled


def visualize_events(raw_data, time, data_root):
    """Visualize events
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        EEG data
    time : str
        Time identifier
    data_root : str
        Data root directory
    """
    print("\n" + "="*80)
    print("Visualizing all events...")
    print("="*80 + "\n")
    
    # Extract events
    events, event_dict = mne.events_from_annotations(raw_data, verbose=True)
    
    # Print event information
    print(f"Total events: {len(events)}")
    print(f"Event types: {event_dict}")
    
    # Create figures directory if not exists
    save_dir = os.path.join(data_root, 'Figures', time)
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot events
    fig = mne.viz.plot_events(
        events, 
        event_id=event_dict, 
        sfreq=raw_data.info['sfreq'],
        show=False
    )
    
    # Adjust figure size for better visualization
    fig.set_size_inches(12, 6)
    fig.suptitle(f'All Events', fontsize=14)
    
    # Display figure
    plt.show()
    
    print("\n" + "="*80)
    print("Event visualization complete")
    print("="*80 + "\n")

def process_events(raw_data, keep_descriptions=['1', '2', '4', '5', '6', '7'], keep_last_n=126):
    """Process events in Raw object, only keep specified event IDs and filter last N events
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw object containing events
    keep_descriptions : list
        Event description list to keep, default ['1', '2', '4', '5', '6', '7']
    keep_last_n : int
        Number of last N events to keep, default 126 (3 each for '1','2' resting events, 30 each for '4','5','6','7' motor imagery events)
        
    Returns
    -------
    mne.io.Raw
        Updated Raw object with filtered events
    event_id : dict
        Event ID dictionary containing descriptions and corresponding event IDs
    """
    import mne
    import numpy as np
    
    # Extract events from Raw object annotations
    events, event_id = mne.events_from_annotations(raw_data)
    
    # Create copy of Raw object
    new_raw = raw_data.copy()
    
    # Direct approach: filter annotations directly
    annotations = raw_data.annotations
    
    # Filter descriptions
    mask = np.zeros(len(annotations.description), dtype=bool)
    for desc in keep_descriptions:
        mask = mask | (annotations.description == desc)
    
    filtered_onset = annotations.onset[mask]
    filtered_duration = annotations.duration[mask]
    filtered_description = annotations.description[mask]
    
    # Only keep last N
    if len(filtered_onset) > keep_last_n:
        filtered_onset = filtered_onset[-keep_last_n:]
        filtered_duration = filtered_duration[-keep_last_n:]
        filtered_description = filtered_description[-keep_last_n:]
    
    # Create new annotations
    new_annotations = mne.Annotations(
        onset=filtered_onset,
        duration=filtered_duration,
        description=filtered_description
    )
    
    # Set new annotations
    new_raw.set_annotations(new_annotations)
    
    return new_raw, event_id

def save_epochs_to_mat(epochs, subject_index, time, event_type, event_id, output_base_dir='Data/Dataset v1', label_format='description'):
    """Save MNE Epochs object as mat file
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object (containing only one event type)
    subject_index : int
        Subject number
    time : str
        Time directory name
    event_type : str
        Event type ('resting_state' or 'motor_imagery')
    event_id : dict
        Dictionary mapping event descriptions to IDs
    output_base_dir : str
        Base output directory path
    label_format : str
        Label format, 'description' or 'onehot'
    
    Returns
    -------
    None
    """
    # Build complete output directory path
    output_dir = os.path.join(output_base_dir, str(time), 'EEG')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define channels to drop
    channels_to_drop = ['HEO', 'VEO', 'EKG', 'EMG', 'Trigger']
    
    # Drop unwanted channels
    epochs = epochs.drop_channels(channels_to_drop)
    
    # Get channel names
    ch_names = epochs.ch_names
    
    # Set time window based on event type
    if event_type == 'resting_state':
        tmin, tmax = 0, 60
        file_prefix = 'resting_state_events'
    elif event_type == 'motor_imagery':
        tmin, tmax = -5, 20
        file_prefix = 'motor_imagery_events'
    else:
        raise ValueError("event_type must be 'resting_state' or 'motor_imagery'")
    
    # Extract data
    data = epochs.get_data(tmin=tmin, tmax=tmax)
    # Convert data from (trial, channel, time) to (channel, time, trial)
    data = np.transpose(data, (1, 2, 0))
    
    # Get event ids and convert to labels
    if label_format == 'description':
        # Convert to event descriptions
        labels = np.array([event_id[label] for label in epochs.events[:, -1]], dtype=object)
    elif label_format == 'onehot':
        # Print debug information
        print("Event ID dictionary content:", event_id)
        print("Event IDs in epochs:", np.unique(epochs.events[:, -1]))
        
        # Use event IDs in epochs directly as categories
        unique_events = sorted(set(epochs.events[:, -1]))
        # Create event ID to index mapping
        event_to_idx = {event: idx for idx, event in enumerate(unique_events)}
        # Get index for each event
        event_indices = [event_to_idx[event] for event in epochs.events[:, -1]]
        # Create one-hot encoding (n_classes, n_events)
        n_events = len(epochs.events)
        n_classes = len(unique_events)
        labels = np.zeros((n_classes, n_events))
        for i, idx in enumerate(event_indices):
            labels[idx, i] = 1
    else:
        raise ValueError("label_format must be 'description' or 'onehot'")
    
    # Pad subject_index to two digits
    subject_str = f"{subject_index:02d}"
    
    # Save data
    if len(data) > 0:
        data_dict = {
            'data': np.array(data),
            'label': labels,
            'ch_names': np.array(ch_names, dtype=object),
        }
        file_name = os.path.join(output_dir, f'subject_{subject_str}_{file_prefix}.mat')
        savemat(file_name, data_dict)
        print(f"{event_type} event data saved to: {file_name}")