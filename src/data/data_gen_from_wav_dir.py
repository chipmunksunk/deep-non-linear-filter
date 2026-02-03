import glob
import os
import h5py
import json
import random
import numpy as np
import librosa

# Path where to save the simulated data
SOURCE_DATA_PATH = "/home/hahmad/Documents/MATLAB/LTSforWASN-main/databases/2025_12_02_15_33_55_nSetups20_15dB_400ms"
DEST_DATA_PATH = "./data/prep/"

def prep_speaker_mix_data_from_wav_dir(source_dataset_dir: str,
                          store_dir: str, 
                          post_fix: str = None, 
                          num_files: dict = {'train': -1, 'val': -1, 'test': -1},
                          target_fs: int = 16000,
                          target_utterance_length_s: int = 10,
                          ):
    """
    Custom preparation of dataset from a source wav directory containing subdirectories 'train', 'validate' and 'test' with multi-channel wav files for 'total', 'clean_speech' and 'noise' signals.

    :param store_dir: path to directory in which to store the dataset
    :param post_fix: postfix to specify the characteristics of the dataset
    :return:
    """
    source_wav_dir = os.path.join(source_dataset_dir, 'wav_files')
    source_meta_dir = os.path.join(source_dataset_dir, 'mat_files')
    post_fix = os.path.split(source_dataset_dir)[-1] if post_fix is None else post_fix

    prep_store_name = f"prep_mix{'_' + post_fix if post_fix else ''}.hdf5"
    
    meta = {}

    with h5py.File(os.path.join(store_dir, prep_store_name), 'w') as prep_storage:
        for dataset_name in ['train', 'val', 'test']:
            # if desired number of files is zero skip dataset creation
            if num_files[dataset_name] == 0:
                continue

            n_dataset_samples = len(list(sorted(glob.glob(os.path.join(source_wav_dir, dataset_name, '*total.wav')))))
            
            # determine number of files to create
            n_dataset_samples = num_files[dataset_name] if num_files[dataset_name] > 0 else n_dataset_samples

            # pre-loading a single sample to determine number of channels
            n_channels = librosa.load(os.path.join(source_wav_dir, dataset_name, 'setup_1_total.wav'), mono=False, sr=target_fs)[0].shape[0]  

            MAX_SAMPLES_PER_FILE = target_utterance_length_s * target_fs
            
            audio_dataset = prep_storage.create_dataset(dataset_name,
                                                        shape=(
                                                            n_dataset_samples, 3, n_channels, MAX_SAMPLES_PER_FILE),
                                                        chunks=(
                                                            1, 3, n_channels, MAX_SAMPLES_PER_FILE),
                                                        dtype=np.float32,
                                                        compression="gzip",
                                                        shuffle=True)

            # meta data dictionary which containts for each sample the RT60, room dimensions, microphone positions (with rotations),
            # the target file name, number of samples, target position and angle and interfering speaker file names and positions
            dataset_meta = {}

            for sample_idx in range(n_dataset_samples):
                sample_meta = get_meta_from_json(os.path.join(source_meta_dir, dataset_name, f'setup_{sample_idx+1}_meta.json'))
                reverb_target_signal = librosa.load(os.path.join(source_wav_dir, dataset_name, f'setup_{sample_idx+1}_speech.wav'), sr=target_fs, mono=False)[0]
                noise_signal = librosa.load(os.path.join(source_wav_dir, dataset_name, f'setup_{sample_idx+1}_noise.wav'), sr=target_fs, mono=False)[0]
                dry_target_signal = np.expand_dims(librosa.load(os.path.join(source_wav_dir, dataset_name, f'setup_{sample_idx+1}_clean_speech.wav'), sr=target_fs, mono=False)[0], axis=0)

                for idx, audio_signal in enumerate([reverb_target_signal, noise_signal, dry_target_signal]):
                    n_audio_samples = min(
                        audio_signal.shape[-1], MAX_SAMPLES_PER_FILE)
                    audio_dataset[sample_idx, idx, :, :n_audio_samples] = audio_signal[:, :n_audio_samples]
                    audio_dataset[sample_idx, idx, :, n_audio_samples:MAX_SAMPLES_PER_FILE] = 0

                dataset_meta[sample_idx] = sample_meta

            meta[dataset_name] = dataset_meta

    with open(os.path.join(store_dir, f"prep_mix_meta{'_' + post_fix if post_fix else ''}.json"),
              'w') as prep_meta_storage:
        json.dump(meta, prep_meta_storage, indent=4)

def get_meta_from_json(json_path:str):
    """
    Load meta data from JSON file.

    :param json_path: path to JSON file
    :return: meta data dictionary
    """
    meta = {}

    with open(json_path, 'r') as meta_file:
        meta_json = json.load(meta_file)

    meta["rt"] = meta_json["sampling"]["T60_ms"]
    meta["room_dim"] = meta_json["room_size_m"]
    meta["mic_pos"] = meta_json["positions"]["mics_m"]
    # meta["mic_phi"] = phi
    # meta["target_file"] = speaker_list[0].split("wsj0")[-1].replace("\\", "/")
    meta["n_samples"] = meta_json["sampling"]["fs_Hz"] * meta_json["sampling"]["length_s"]
    meta["target_pos"] = meta_json["positions"]["sources_m"]
    # meta["target_angle"] = target_angle
    # meta[f"interf{interf_idx}_file"] = interf_path.split("wsj0")[-1].replace("\\", "/")
    # meta[f"interf{interf_idx}_pos"] = interf_source.tolist()
    meta["snr"] = np.mean(meta_json["mic"]["snr_realized_dB"])
    
    return meta

if __name__ == '__main__':
    prep_speaker_mix_data_from_wav_dir(SOURCE_DATA_PATH, DEST_DATA_PATH, num_files= {'train': -1, 'val': -1, 'test': 0})