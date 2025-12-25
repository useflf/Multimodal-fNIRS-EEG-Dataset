import os
import traceback
from utils import *


def preprocess_eeg_data(raw_data, bad_channels, time, data_root, subject_num):
    """Execute complete EEG data preprocessing pipeline
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw EEG data
    bad_channels : list
        List of bad channels
    time : str
        Time identifier
    data_root : str
        Data root directory
    subject_num : int
        Subject index number
    
    Returns
    -------
    mne.Epochs
        Preprocessed epoch data
    """
    # 1. Display raw data information
    display_data_info(raw_data)
    
    # 2. Save electrode position information
    montage_file = os.path.join(data_root, 'FIF_Data\mnt.mat')
    save_electrode_montage(raw_data, montage_file)
    
    # visualize_events(raw_data, time, data_root)
    # 3. Event processing and visualization, keep motor imagery and resting state events
    new_raw, event_id = process_events(raw_data)

    # 4. Bandpass filtering and notch filtering
    raw_filter = apply_filters(new_raw)

    # Bad channel interpolation
    raw_clean = interpolate_bad_channels(raw_filter, bad_channels)

    # 6. Create epochs (includes baseline correction)
    motor_imagery = create_epochs(raw_clean, epoch_type='motor_imagery')
    resting_state = create_epochs(raw_clean, epoch_type='resting_state')

    # 7. Downsampling
    motor_imagery_resampled = resample_epochs(motor_imagery)
    resting_state_resampled = resample_epochs(resting_state)

    print("\n" + "*"*80)
    print(f"All preprocessing tasks completed - Subject {subject_num} for {time} processed")
    print("*"*80 + "\n")
    
    return motor_imagery_resampled, resting_state_resampled, event_id

def main():
    """Main function, includes complete EEG data processing pipeline"""
    # Get current working directory
    data_root = "D:\Code\Dataset_4-classification MI"
    print(f"Current working directory: {data_root}")
    
    # Load configuration file
    info = load_data_from_json(os.path.join(data_root, 'info.json'))
    if info is None:
        print("Unable to load configuration file")
        return
    
    # Iterate through all times in data structure
    for time, data_groups in info.items():
        # Iterate through each data group
        for data_group in data_groups:
            # Iterate through each data element
            for i, item in enumerate(data_group["data"]):
                # Subject number
                subject_num = i + 1
                subject_index = item["data_element"]
                # Get bad channels list
                bad_channels = item["bad_channels"]
                
                # Build file path
                file_paths = [os.path.join(data_root, 'Data', time, 'EEG', f'Acquisition {idx}.dat') for idx in subject_index]
                
                # Load raw EEG data
                try:
                    raw_data = load_raw_eeg_data(file_paths)
                except Exception as e:
                    print(f"Error loading data: {e}")
                    print("Stack trace:")
                    print(traceback.format_exc())
                    continue
                
                # Execute preprocessing
                try:
                    motor_imagery_resampled, resting_state_resampled, event_id = preprocess_eeg_data(
                        raw_data, bad_channels, time, data_root, subject_num
                    )
                    
                    # Save processed data
                    save_epochs_to_mat(motor_imagery_resampled, subject_index=subject_index, 
                                     time=time, event_type='motor_imagery', event_id=event_id, 
                                     output_base_dir=os.path.join(data_root, 'Data', 'Dataset v1'),
                                     label_format='onehot')
                    save_epochs_to_mat(resting_state_resampled, subject_index=subject_index, 
                                     time=time, event_type='resting_state', event_id=event_id, 
                                     output_base_dir=os.path.join(data_root, 'Data', 'Dataset v1'),
                                     label_format='onehot')
                except Exception as e:
                    print(f"Error preprocessing data: {e}")
                    print("Stack trace:")
                    print(traceback.format_exc())
                    continue


if __name__ == "__main__":
    main()
