import torchaudio
import matplotlib.pyplot as plt
import torch


def main(args):
    wav_a, _ = load_audio(args.path_a)
    wav_b, _ = load_audio(args.path_b)

    vocoder = torch.hub.load("tuanio/melgan", "load_melgan")
    def cal_mel(wav):
        return vocoder(wav).detach().cpu()[0]

    fig, ax = plt.subplots(ncols=2, sharey=True)

    ax[0].imshow(cal_mel(wav_a))
    ax[1].imshow(cal_mel(wav_b))

    ax[0].set_title('Original')
    ax[1].set_title('Converted')

    ax[0].grid(alpha=.25, ls='-.', color='black')
    ax[1].grid(alpha=.25, ls='-.', color='black')

    fig.tight_layout()
    plt.savefig(f'{args.name}.png')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-a")
    parser.add_argument("--path-b")
    parser.add_argument('--name')
    args = parser.parse_args()
    main(args)
