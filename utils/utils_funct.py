
import numpy as np
import librosa


melW = librosa.filters.mel(sr=fs,
                           n_fft=nfft,
                           n_mels=mel_bins,
                           fmin=fmin)


def logmel(sig):

    S = np.abs(librosa.stft(y=sig,
                            n_fft=nfft,
                            hop_length=hopsize,
                            center=True,
                            window=window,
                            pad_mode='reflect'))**2
    S_mel = np.dot(melW, S).T
    S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
    S_logmel = np.expand_dims(S_logmel, axis=0)

    return S_logmel


def gcc_phat(sig, refsig):

    ncorr = 2*1024 - 1
    nfft = int(2**np.ceil(np.log2(np.abs(ncorr))))
    Px = librosa.stft(y=sig,
                      n_fft=nfft,
                      hop_length=hopsize,
                      center=True,
                      window=window,
                      pad_mode='reflect')
    Px_ref = librosa.stft(y=refsig,
                          n_fft=nfft,
                          hop_length=hopsize,
                          center=True,
                          window=window,
                          pad_mode='reflect')

    R = Px*np.conj(Px_ref)

    n_frames = R.shape[1]
    gcc_phat = []
    for i in range(n_frames):
        spec = R[:, i].flatten()
        cc = np.fft.irfft(np.exp(1.j*np.angle(spec)))
        cc = np.concatenate((cc[-mel_bins//2:], cc[:mel_bins//2]))
        gcc_phat.append(cc)
    gcc_phat = np.array(gcc_phat)
    gcc_phat = gcc_phat[None, :, :]

    return gcc_phat


def transform(audio):

    channel_num = audio.shape[0]
    feature_logmel = []
    feature_gcc_phat = []
    for n in range(channel_num):
        feature_logmel.append(logmel(audio[n]))
        for m in range(n+1, channel_num):
            feature_gcc_phat.append(
                gcc_phat(sig=audio[m], refsig=audio[n]))

    feature_logmel = np.concatenate(feature_logmel, axis=0)
    feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
    feature = np.concatenate([feature_logmel, feature_gcc_phat])

    return feature


def calculate_scalar(features):

    mean = []
    std = []

    channels = features.shape[0]
    for channel in range(channels):
        feat = features[channel, :, :]
        mean.append(np.mean(feat, axis=0))
        std.append(np.std(feat, axis=0))

    mean = np.array(mean)
    std = np.array(std)
    mean = np.expand_dims(mean, axis=0)
    std = np.expand_dims(std, axis=0)
    mean = np.expand_dims(mean, axis=2)
    std = np.expand_dims(std, axis=2)

    return mean, std


def transforms(x, mean, std):
    """

    Use the calculated scalar to transform data.
    """

    return (x - mean) / std
