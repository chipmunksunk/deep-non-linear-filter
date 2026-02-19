import glob
import os
import h5py
import json
import random
import numpy as np
import librosa
import shutil

SOURCE_DATA_PATH = "/home/hahmad/Documents/MATLAB/LTSforWASN-main/databases/2026_02_18_15_01_49_nSetups20_10dB_whiteNoise"
DEST_DATA_PATH = "./src/data/prep/"
DEST_RAW_DATA_PATH = "./src/data/raw/"
# PFIX = {"clean": "clean_speech.wav", "noise": "noise_hat_sc.wav", "reverb": "d_hat_sc.wav", "meta": "meta.json"}
PFIX = {"clean": "cleanSpeech.wav", "noise": "noise.wav", "reverb": "micSpeech.wav", "meta": "meta.json"}

def prep_speaker_mix_data_from_wav_dir(source_dataset_dir: str,
                          store_dir: str, 
                          dataset_id: str = None, 
                          num_files: dict = {'train': -1, 'val': -1, 'test': -1},
                          target_fs: int = 16000,
                          target_utterance_length_s: int = 10,
                          ):
    """
    Custom preparation of dataset from a source wav directory containing subdirectories 'train', 'validate' and 'test' with multi-channel wav files for 'total', 'clean_speech' and 'noise' signals.

    :param store_dir: path to directory in which to store the dataset
    :param dataset_id: postfix to specify the characteristics of the dataset
    :return:
    """
    dataset_id = os.path.split(source_dataset_dir)[-1] if dataset_id is None else dataset_id

    prep_store_name = f"prep_mix{'_' + dataset_id if dataset_id else ''}.hdf5"
    
    meta = {}

    with h5py.File(os.path.join(store_dir, prep_store_name), 'w') as prep_storage:
        for dataset_name in ['train', 'val', 'test']:
            # if desired number of files is zero skip dataset creation
            if num_files[dataset_name] == 0:
                continue
            
            files_meta = [name for name in os.listdir(os.path.join(source_dataset_dir, dataset_name)) if name.endswith(PFIX["meta"])]
            sample_ids = [name.split("_")[1] for name in files_meta]
            n_dataset_samples = len(files_meta)
            
            # determine number of files to create
            n_dataset_samples = num_files[dataset_name] if num_files[dataset_name] > 0 else n_dataset_samples

            # pre-loading a single sample to determine number of channels
            temp_signal = librosa.load(os.path.join(source_dataset_dir, dataset_name, f'setup_{sample_ids[0]}_{PFIX["reverb"]}'), mono=False, sr=target_fs)[0]
            n_channels = temp_signal.shape[0] if len(temp_signal.shape) > 1 else 1

            MAX_SAMPLES_PER_FILE = target_utterance_length_s * target_fs
            
            # prepare hdf5 dataset with shape (n_samples, 3, n_channels, max_samples_per_file) for the reverberant target signal, noise signal and dry target signal
            audio_dataset = prep_storage.create_dataset(dataset_name,
                                                        shape=(
                                                            n_dataset_samples, 3, n_channels, MAX_SAMPLES_PER_FILE),
                                                        chunks=(
                                                            1, 3, n_channels, MAX_SAMPLES_PER_FILE),
                                                        dtype=np.float32,
                                                        compression="gzip",
                                                        shuffle=True)

            # meta data dictionary which containts for each sample the RT60, room dimensions, microphone positions (with rotations),
            # the target file name, number of samples, target position and angle, and interfering speaker file names and positions
            dataset_meta = {}

            # for sample_idx, (meta_file, reverb_file, noise_file, clean_file) in enumerate(zip(files_meta, files_reverb_speech, files_noise, files_clean_speech)):
            for idx, sample_id in enumerate(sample_ids):
                sample_meta  = get_meta_from_json(os.path.join(source_dataset_dir, dataset_name, f'setup_{sample_id}_{PFIX["meta"]}'))
                reverb_target_signal = librosa.load(os.path.join(source_dataset_dir, dataset_name, f'setup_{sample_id}_{PFIX["reverb"]}'), sr=target_fs, mono=False)[0]
                noise_signal = librosa.load(os.path.join(source_dataset_dir, dataset_name, f'setup_{sample_id}_{PFIX["noise"]}'), sr=target_fs, mono=False)[0]
                dry_target_signal = librosa.load(os.path.join(source_dataset_dir, dataset_name, f'setup_{sample_id}_{PFIX["clean"]}'), sr=target_fs, mono=False)[0]

                for audio_type_idx, audio_signal in enumerate([reverb_target_signal, noise_signal, dry_target_signal]):
                    n_audio_samples = min(
                        audio_signal.shape[-1], MAX_SAMPLES_PER_FILE)
                    audio_dataset[idx, audio_type_idx, :, :n_audio_samples] = audio_signal[...,:n_audio_samples]
                    audio_dataset[idx, audio_type_idx, :, n_audio_samples:MAX_SAMPLES_PER_FILE] = 0

                dataset_meta[idx] = sample_meta

            meta[dataset_name] = dataset_meta

    with open(os.path.join(store_dir, f"prep_mix_meta{'_' + dataset_id if dataset_id else ''}.json"),
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

    # meta["rt"] = meta_json["sampling"]["T60_ms"]
    # meta["room_dim"] = meta_json["room_size_m"]
    meta["mic_pos"] = meta_json["positions"]["mics_m"]
    # meta["mic_phi"] = phi
    # meta["target_file"] = speaker_list[0].split("wsj0")[-1].replace("\\", "/")
    # meta["n_samples"] = meta_json["sampling"]["fs_Hz"] * meta_json["sampling"]["length_s"]

    # TODO: Hardcoded target utterance length for now, will be changed when json files are adjusted
    meta["n_samples"] = 16000*10
    meta["target_pos"] = meta_json["positions"]["sources_m"]

    # meta["target_angle"] = target_angle
    # meta[f"interf{interf_idx}_file"] = interf_path.split("wsj0")[-1].replace("\\", "/")
    # meta[f"interf{interf_idx}_pos"] = interf_source.tolist()
    meta["snr"] = np.mean(meta_json["mic"]["snr_realized_dB"])
    
    return meta

def import_data(source_dataset_dir:str, dest_dataset_dir:str, substring:list = None, overwrite: bool = False):
    """
    Import data from source dataset directory and store it in destination raw dataset directory.

    :param source_dataset_dir: path to source dataset directory
    :param dest_dataset_dir: path to destination dataset directory
    :param substring: substring or list of substrings to filter files to be imported. If empty, all files are imported.
    :param overwrite: whether to overwrite existing files in destination directory
    :return:
    """
    if overwrite or substring is None:
        Warning(f"Either overwrite is set to True or no substring is given. Existing files in {dest_dataset_dir} will be overwritten.")
        wait = input("Do you want to continue? (y/n): ")
        if wait.lower() != 'y':
            print("Aborting data import.")
            return
        
    if isinstance(substring, str):
        substring = [substring]
    
    if substring is None:
        shutil.copytree(source_dataset_dir, dest_dataset_dir, dirs_exist_ok=True)
        copy_counter = "All"
    else:
        copy_counter = 0
        for root, dirs, files in os.walk(source_dataset_dir):
            for file in files:
                if any(sub in file for sub in substring):
                    source_file_path = os.path.join(root, file)
                    dest_file_path = os.path.join(dest_dataset_dir, file)
                    if overwrite or not os.path.exists(dest_file_path):
                        shutil.copy(source_file_path, dest_file_path)
                        copy_counter += 1

    print("Data import completed. {} files copied from {} to {}.".format(copy_counter, source_dataset_dir, dest_dataset_dir))

def create_train_val_test_split(raw_dataset_dir:str, split_ratios: dict = {'train': 0.8, 'val': 0.1, 'test': 0.1}):
    """
    Create train, validation and test split from raw dataset directory containing wav files and corresponding meta JSON files.
    The wav files are expected to be named in the following format: 
    
    dry target signal:          'setup_{idx}_clean_speech.wav'
    noise signal:               'setup_{idx}_noise_hat_sc.wav'
    reverberant target signal:  'setup_{idx}_d_hat_sc.wav'
    
    , where {idx} is the running index for each sample. 
    
    The corresponding meta JSON files are expected to be named in the format 'setup_{idx}_meta.json'.
    
    :param raw_dataset_dir: path to raw dataset directory
    :param split_ratios: split ratios for train, validation and test set. The values should sum up to 1. The keys should be 'train', 'val' and 'test'.
    """
    # identify postfixes of the files corresponding to the dry target signal, noise signal, reverberant target signal and meta JSON files, respectively.
    global PFIX

    # check whether train, validation and test split already exist and ask for confirmation to overwrite
    split_dirs = ['train', 'val', 'test']

    if any(os.path.exists(os.path.join(raw_dataset_dir, split_dir)) for split_dir in split_dirs):
        print(f"Either some or all split folders already exist in {raw_dataset_dir}. Existing directories will be emptied.")
        wait = input("Do you want to continue? (y/n): ")
        if wait.lower() != 'y':
            print("Aborting train, validation and test split creation.")
            return
        [shutil.rmtree(os.path.join(raw_dataset_dir, split_dir)) for split_dir in split_dirs]
        
    # check if split ratios are compute number of samples for each split
    if sum(split_ratios.values()) != 1:
        raise ValueError("The values of split_ratios should sum up to 1.")
    numSamples = len([name for name in os.listdir(raw_dataset_dir) if name.endswith(".json")])
    nTrain = int(numSamples * split_ratios['train'])
    nVal = int(numSamples * split_ratios['val'])
    nTest = numSamples - nTrain - nVal

    # shuffle json files and split into train, validation and test set
    indices_list = list(range(1,numSamples+1))
    random.shuffle(indices_list)

    train_files = indices_list[:nTrain]
    val_files = indices_list[nTrain:nTrain+nVal]
    test_files = indices_list[nTrain+nVal:]

    os.makedirs(os.path.join(raw_dataset_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(raw_dataset_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(raw_dataset_dir, "test"), exist_ok=True)

    for idx in train_files:
        for post_fix in PFIX.values():
            f = f"setup_{idx}_{post_fix}"
            shutil.move(os.path.join(raw_dataset_dir, f), os.path.join(raw_dataset_dir, "train", f))

    for idx in val_files:
        for post_fix in PFIX.values():
            f = f"setup_{idx}_{post_fix}"
            shutil.move(os.path.join(raw_dataset_dir, f), os.path.join(raw_dataset_dir, "val", f))

    for idx in test_files:
        for post_fix in PFIX.values():
            f = f"setup_{idx}_{post_fix}"
            shutil.move(os.path.join(raw_dataset_dir, f), os.path.join(raw_dataset_dir, "test", f))

    print(f"Train, validation and test split created with {nTrain} training samples, {nVal} validation samples and {nTest} test samples.")

if __name__ == '__main__':
    # import_data(os.path.join(SOURCE_DATA_PATH, 'processed', 'best', 'gevd_ml'), DEST_RAW_DATA_PATH, substring=[PFIX["noise"], PFIX["reverb"]])
    import_data(os.path.join(SOURCE_DATA_PATH, 'wav_files'), DEST_RAW_DATA_PATH, substring=[PFIX["noise"], PFIX["reverb"]])
    import_data(os.path.join(SOURCE_DATA_PATH, 'wav_files'), DEST_RAW_DATA_PATH, substring=[PFIX["clean"]])
    import_data(os.path.join(SOURCE_DATA_PATH, 'mat_files'), DEST_RAW_DATA_PATH, substring=[PFIX["meta"]])

    create_train_val_test_split(DEST_RAW_DATA_PATH, split_ratios={'train': 0.8, 'val': 0.1, 'test': 0.1})
    prep_speaker_mix_data_from_wav_dir(DEST_RAW_DATA_PATH, DEST_DATA_PATH, dataset_id=os.path.split(SOURCE_DATA_PATH)[-1], num_files= {'train': -1, 'val': -1, 'test': -1}, target_fs=16000, target_utterance_length_s=10) 