import argparse
from args.base_arg_parser import BaseArgParser
from pymcd.mcd import Calculate_MCD
from utils import get_mcd_calculator
from tqdm.auto import tqdm
import glob


def main():
    calculate_mcd = get_mcd_calculator()

    project_audio_path = args.project_audio_path
    for folder in glob.glob(project_audio_path + "/*"):
        project_name = folder.rsplit(os.sep, 1)[-1]
        audios_path = os.path.join(folder, "converted_audio")
        original_files = sorted(glob.glob(audios_path + "/*_original_*.wav"))
        converted_files = sorted(glob.glob(audios_path + "/*_converted_*.wav"))

        mcd_scores = []
        for orig, conv in tqdm(zip(original_files, converted_files), total=len(original_files)):
            mcd_score = calculate_mcd(orig, conv)
            mcd_scores.append(mcd_score)

        mcd_scores = np.array(mcd_scores)
        m = np.mean(mcd_scores)
        s = np.std(mcd_scores)
        print(f"Project: [{project_name}] | [{np.round(m, 2)}] +- [{np.round(s, 2)}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-audio-path")
    args = parser.parse_args()
    main(args)
