import os
import glob
import json
import argparse
import numpy as np
from tqdm.auto import tqdm
from args.base_arg_parser import BaseArgParser
from pymcd.mcd import Calculate_MCD
from mask_cyclegan_vc.utils import get_mcd_calculator


def main(args):
    calculate_mcd = get_mcd_calculator()

    list_mcd_scores = []
    project_audio_path = args.project_audio_path
    for folder in glob.glob(project_audio_path + "/*"):
        project_name = folder.rsplit(os.sep, 1)[-1]
        audios_path = os.path.join(folder, "converted_audio")
        print("Audio path:", audios_path)
        original_files = sorted(glob.glob(audios_path + "/*-original_*.wav"))
        converted_files = sorted(glob.glob(audios_path + "/*-converted_*.wav"))

        # print(f"Original files: {len(original_files)}, Converted files: {len(converted_files)}")
        mcd_scores = []
        for orig, conv in tqdm(zip(original_files, converted_files), total=len(original_files)):
            mcd_score = calculate_mcd(orig, conv)
            mcd_scores.append(mcd_score)
            is_freq_mask = 'FreqMask20' in project_name
            list_mcd_scores.append((mcd_score, is_freq_mask, orig, conv))

        mcd_scores = np.array(mcd_scores)
        m = np.mean(mcd_scores)
        s = np.std(mcd_scores)
        print(f"Project: [{project_name}] | MCD = [{np.round(m, 2)}] +- [{np.round(s, 2)}]")

    sorted_list = sorted(list_mcd_scores)
    with open('mcd_score.json', 'w', encoding='utf-8') as f:
        json.dump(sorted_list, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-audio-path")
    args = parser.parse_args()
    main(args)
