from argparse import ArgumentParser
from data.datamodule import HDF5DataModule
import yaml
from train_jnf import load_model
import os
import glob
import soundfile as sf

def evaluate_trained_model(log_dir: str, utt_idx: int = 0):
    files = glob.glob(os.path.join(log_dir, 'checkpoints', '*.ckpt'))
    CHECKPOINT_FILE = files[0] if len(files) > 0 else None

    with open(f'{log_dir}/config_used.yaml', 'r') as config_file: 
            config = yaml.safe_load(config_file)

    data_config = config['data']
    dm = HDF5DataModule(**data_config)

    ckpt_file = CHECKPOINT_FILE
    if not ckpt_file is None:
        exp = load_model(ckpt_file, config)
    else:
        raise ValueError("No checkpoint file specified for evaluation. Please provide a checkpoint file in the config under training.resume_ckpt")

    #forward pass of a single utterance
    clean_td, noise_td, est_clean_td, est_noise_td = exp.evaluate_single_utterance(dm.train_dataset, idx=utt_idx)

    #save estimated clean time-domain signal for listening
    sf.write('src/audio/clean.wav', clean_td.detach().cpu().numpy().ravel(), dm.fs)
    sf.write('src/audio/noise.wav', noise_td.detach().cpu().numpy().ravel(), dm.fs)
    sf.write('src/audio/estimated_clean.wav', est_clean_td.detach().cpu().numpy().ravel(), dm.fs)
    sf.write('src/audio/estimated_noise.wav', est_noise_td.detach().cpu().numpy().ravel(), dm.fs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--utt_idx", default=0, type=int)
    args = parser.parse_args()

    # evaluate_trained_model(log_dir=args.logdir, utt_idx=args.utt_idx)
    evaluate_trained_model(log_dir='./logs/tb_logs/JNF/version_11', utt_idx=0)