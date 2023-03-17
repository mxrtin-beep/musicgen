from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import torch
from scipy import signal
from scipy.fft import fftshift
import scipy

import librosa
import numpy as np

from playsound import playsound


'''
wavfile.write('re1.wav', fs, sig3)
playsound('re1.wav')

'''
sig, fs = librosa.load('input.wav')

def get_log_spec(y):

	S = np.abs(librosa.stft(y))
	logS = np.log(S)

	return logS



def get_sig_from_log_spec(log_spec):
	spec = np.exp(log_spec)
	y_inv = librosa.griffinlim(spec)
	return y_inv


logS = get_log_spec(sig)

def plot_logS(logS):
	plt.imshow(logS, origin='lower')
	plt.xlabel('Time')
	plt.ylabel('Frequency')
	plt.savefig("image")


y_inv = get_sig_from_log_spec(logS)

wavfile.write('re2.wav', fs, y_inv)



