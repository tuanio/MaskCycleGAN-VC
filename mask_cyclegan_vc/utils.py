"""
Defines the util functions associated with the cycleGAN VC pipeline.
"""

import io
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import ToTensor

from pymcd.mcd import Calculate_MCD
import pyworld
import pysptk

import librosa
import librosa.display

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def decode_melspectrogram(vocoder, melspectrogram, mel_mean, mel_std):
    """Decoded a Mel-spectrogram to waveform using a vocoder.

    Args:
        vocoder (torch.nn.module): Vocoder used to decode Mel-spectrogram
        melspectrogram (torch.Tensor): Mel-spectrogram to be converted
        mel_mean ([type]): Mean of the Mel-spectrogram for denormalization
        mel_std ([type]): Standard Deviations of the Mel-spectrogram for denormalization

    Returns:
        torch.Tensor: decoded Mel-spectrogram
    """
    denorm_converted = melspectrogram * mel_std + mel_mean
    rev = vocoder.inverse(denorm_converted.unsqueeze(0))
    return rev


def get_mel_spectrogram_fig(spec, title="Mel-Spectrogram"):
    """Generates a figure of the Mel-spectrogram and converts it to a tensor.

    Args:
        spec (torch.Tensor): Mel-spectrogram
        title (str, optional): Figure name. Defaults to "Mel-Spectrogram".

    Returns:
        torch.Tensor: Figure as tensor
    """
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)
    S_db = librosa.power_to_db(10 ** spec.numpy().squeeze(), ref=np.max)
    img = librosa.display.specshow(S_db, ax=ax, y_axis="log", x_axis="time")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)

    image = Image.open(buf)
    image = ToTensor()(image)

    plt.close(fig)
    return image


def new_wav2mcep_numpy(self, loaded_wav, alpha=0.65, fft_size=1024):

    # Use WORLD vocoder to spectral envelope
    _, sp, _ = pyworld.wav2world(
        loaded_wav.astype(np.double),
        fs=self.SAMPLING_RATE,
        frame_period=self.FRAME_PERIOD,
        fft_size=fft_size,
    )

    # Extract MCEP features
    mcep = pysptk.sptk.mcep(
        sp, order=35, alpha=alpha, maxiter=0, etype=1, eps=1.0e-8, min_det=0.0, itype=3
    )

    return mcep


def get_mcd_calculator(mcd_mode="plain"):
    mcd_toolbox = Calculate_MCD(MCD_mode=mcd_mode)
    mcd_toolbox.wav2mcep_numpy = new_wav2mcep_numpy.__get__(mcd_toolbox, Calculate_MCD)
    return lambda path_a, path_b: mcd_toolbox.calculate_mcd(path_a, path_b)
