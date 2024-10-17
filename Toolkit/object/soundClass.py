import torchaudio
from calculate.utils import Uility
from pathlib import Path

mel_converter = torchaudio.transforms.MelSpectrogram(n_mels=80)
db_converter = torchaudio.transforms.AmplitudeToDB()

class SoundClass(Uility):
    @classmethod
    def getSound(cls, path, suffix=".wav"):
        try:
            waveform, sample_rate = torchaudio.load(Path(path).with_suffix('.wav'))
            audio_sample = F.resample(waveform[0], sample_rate, 16000, lowpass_filter_width=6)
            feature = db_converter(mel_converter(audio_sample))
        except Exception as e:
            feature = None
            audio_sample = None
            print(e)
        return audio_sample, feature