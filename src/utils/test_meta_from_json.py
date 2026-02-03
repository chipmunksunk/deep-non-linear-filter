from data.data_gen_from_wav_dir import get_meta_from_json
import os
import json

if __name__ == '__main__':
    json_path = '/home/hahmad/Documents/MATLAB/LTSforWASN-main/databases/2025_12_02_15_33_55_nSetups20_15dB_400ms/mat_files/setup_1_meta.json'

    sample_meta = get_meta_from_json(json_path=json_path)
    
    # with open(json_path, 'r') as meta_file:
    #     meta_json = json.load(meta_file)

    # print(meta_json.keys())

    print(json.dumps(sample_meta, indent=4))