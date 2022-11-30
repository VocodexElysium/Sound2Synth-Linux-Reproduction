from .basic_utils import *
import torch
import torchaudio.functional as F
import torchaudio.transforms as T

def AudioToSpec(
    audio,
    n_fft       = get_config("default_torchaudio_args")["n_fft"],
    win_length  = get_config("default_torchaudio_args")["win_length"],
    hop_length  = get_config("default_torchaudio_args")["hop_length"],
    power       = get_config("default_torchaudio_args")["power"],
    center      = get_config("default_torchaudio_args")["center"],
    pad_mode    = get_config("default_torchaudio_args")["pad_mode"],
):
    """Convert an audio file to a spectrogram via STFT.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> spectrogram = AudioToSpec(waveform)
        >>> waveform
            tensor([[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]])
        >>> spectrogram
            tensor([[[2.3777e-05, 9.9391e-06, 9.6153e-08,  ..., 1.4636e-07,
                      4.5998e-09, 9.1016e-09],
                      ...,
                     [1.9494e-09, 2.0759e-08, 6.0320e-08,  ..., 1.4949e-10,
                      7.1396e-09, 7.3691e-09]]])
    """
    return T.Spectrogram(
        n_fft = n_fft,
        win_length = win_length,
        hop_length = hop_length,
        power = power,
        center = center,
        pad_mode = pad_mode,
    )(audio)

def SpecToAudio(
    spectrogram,
    n_fft       = get_config("default_torchaudio_args")["n_fft"],
    win_length  = get_config("default_torchaudio_args")["win_length"],
    hop_length  = get_config("default_torchaudio_args")["hop_length"],
    power       = get_config("default_torchaudio_args")["power"],
):
    """Convert a spectrogram to an audio via the Griffin-Lim transformation.
    
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> spectrogram = AudioToSpec(waveform)
        >>> waveform
            tensor([[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]])
        >>> spectrogram
            tensor([[[2.3777e-05, 9.9391e-06, 9.6153e-08,  ..., 1.4636e-07,
                      4.5998e-09, 9.1016e-09],
                      ...,
                     [1.9494e-09, 2.0759e-08, 6.0320e-08,  ..., 1.4949e-10,
                      7.1396e-09, 7.3691e-09]]])
        >>> waveform_hat = SpecToAudio(spectrogram)
        >>> waveform_hat
        tensor([[ 3.9386e-02,  2.2363e-02,  1.4716e-02,  ..., -2.9483e-06,
                 -4.3291e-06, -3.0011e-06],
                [ 1.5335e-02,  7.6627e-03,  2.1451e-02,  ..., -1.1247e-07,
                 -3.5824e-06,  1.3104e-06]])
    """
    return T.GriffinLim(
        n_fft = n_fft,
        win_length = win_length,
        hop_length = hop_length,
        power = power,
    )(spectrogram)

def AudioToMel(
    audio,
    sample_rate = get_config("default_torchaudio_args")['sample_rate'],
    n_fft       = get_config("default_torchaudio_args")["n_fft"],
    win_length  = get_config("default_torchaudio_args")["win_length"],
    hop_length  = get_config("default_torchaudio_args")["hop_length"],
    power       = get_config("default_torchaudio_args")["power"],
    center      = get_config("default_torchaudio_args")["center"],
    pad_mode    = get_config("default_torchaudio_args")["pad_mode"],
    norm        = get_config("default_torchaudio_args")["norm"],
    onesided    = get_config("default_torchaudio_args")["onesided"],
    n_mels      = get_config("default_torchaudio_args")["n_mels"],
    mel_scale   = get_config("default_torchaudio_args")["mel_scale"],
):
    """Convert an audio to a melspectrogram via FFT.
    
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> melspectrogram = AudioToMel(waveform)
        >>> waveform
            tensor([[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]])
        >>> melspectrogram
        tensor([[[2.4874e-07, 3.0650e-07, 6.1774e-08,  ..., 1.3627e-09,
                  1.3937e-09, 5.3404e-10],
                  ...,
                 [4.5732e-09, 3.7288e-09, 2.5635e-09,  ..., 1.9378e-10,
                  3.5321e-10, 1.2057e-10]]])
    """
    return T.MelSpectrogram(
        sample_rate = sample_rate,
        n_fft = n_fft,
        win_length = win_length,
        hop_length = hop_length,
        power = power,
        center = center,
        pad_mode = pad_mode,
        norm = norm,
        onesided = onesided,
        n_mels = n_mels,
        mel_scale = mel_scale
    )(audio)

def AudioToMFCC(
    audio,
    sample_rate = get_config("default_torchaudio_args")['sample_rate'],
    n_mfcc      = get_config("default_torchaudio_args")['n_mfcc'],
    n_fft       = get_config("default_torchaudio_args")["n_fft"],
    win_length  = get_config("default_torchaudio_args")["win_length"],
    hop_length  = get_config("default_torchaudio_args")["hop_length"],
    power       = get_config("default_torchaudio_args")["power"],
    center      = get_config("default_torchaudio_args")["center"],
    pad_mode    = get_config("default_torchaudio_args")["pad_mode"],
    norm        = get_config("default_torchaudio_args")["norm"],
    onesided    = get_config("default_torchaudio_args")["onesided"],
    n_mels      = get_config("default_torchaudio_args")["n_mels"],
    mel_scale   = get_config("default_torchaudio_args")["mel_scale"],
):
    """Convert an audio to n_mfcc MFCC components, which calculates the MFCC on the dB-scale melspectrogram by default. 
    
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> mfcc = AudioToMFCC(waveform)
        >>> waveform
            tensor([[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]])
        >>> mfcc
        tensor([[[-2.7598e+02, -2.3833e+02, -2.5868e+02,  ..., -6.7007e+02,
                  -6.7007e+02, -6.7007e+02],
                  ...,
                 [-1.6160e+01, -3.2460e+01, -3.4852e+01,  ..., -1.5947e-05,
                  -1.5947e-05, -1.5947e-05]]])
    """
    return T.MFCC(
        sample_rate = sample_rate,
        n_mfcc = n_mfcc,
        melkwargs = {
            "n_fft"      : n_fft,
            "win_length" : win_length,
            "hop_length" : hop_length,
            "power"      : power,
            "center"     : center,
            "pad_mode"   : pad_mode,
            "norm"       : norm,
            "onesided"   : onesided,
            "n_mels"     : n_mels,
            "mel_scale"  : mel_scale,
        }
    )(audio)

def TrimAudio(audio, trim_scale=1e-5):
    """Cut the empty part from the tail of the audio.
    
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> trimed_audio = TrimAudio(waveform)
        >>> waveform
            tensor([[0., 0., 0.,  ..., 0., 0., 0.],
                    [0., 0., 0.,  ..., 0., 0., 0.]])
        >>> trimed_audio
            tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
                      0.0000e+00,  0.0000e+00],
                    [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ..., -3.0518e-05,
                     -3.0518e-05, -3.0518e-05]])
    """
    #Stereo audio file will result an array of shape [2, L] in torchaudio.
    new_audio = audio.T
    
    trim_pos = len(new_audio) - 1
    while trim_pos > 0 and (new_audio[trim_pos] ** 2).mean() <= trim_scale ** 2:
        trim_pos -= 1
    new_audio = new_audio[:trim_pos + 1, :]
    return new_audio.T

def PadAudio(audio, target_length):
    """Pad the audio into the target_length.
    
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> padded_audio = PadAudio(waveform, 307200)
        >>> len(waveform.T)
            259455
        >>> len(padded_audio.T)
            307200
    """
    #Stereo audio file will result an array of shape [2, L] in torchaudio.
    new_audio = audio.T
    audio_length = len(new_audio)

    if audio_length > target_length:
        raise ValueError("The audio is already longer than {0}!".format(audio_length))

    if audio_length < target_length:
        pad = torch.zeros_like(new_audio[0])
        new_audio = torch.cat([
            new_audio,
            torch.stack([pad for _ in range(target_length - audio_length)], dim=0)],
            dim=0
            )

    return new_audio.T

def AdjustAudioLength(audio, target_length, trim_scale=1e-5, force_trim=False):
    """Adjust Audio length to the target length.
       The audio will be trimmed if its length is bigger than the target length and force_trim=True,
       and the audio will be padded to the target length if its length is smaller or equal to the target length.
       
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> adjusted_audio1 = AdjustAudioLength(waveform, 307200)
        >>> len(waveform.T)
            259455
        >>> len(adjusted_audio1.T)
            307200   
        >>> adjusted_audio2 = AdjustAudioLength(waveform, 30720)
            ValueError: The trimmed audio is still too long to adjust to length 30720!
        >>> adjusted_audio2 = AdjustAudioLength(waveform, 30720, force_trim=True)
        >>> len(adjusted_audio2.T)
            30720
    """
    #Stereo audio file will result a array of shape [2, L] in torchaudio.
    new_audio = TrimAudio(audio, trim_scale=trim_scale).T
    if len(new_audio) > target_length and force_trim == True:
        new_audio = new_audio[:target_length, :]
    elif len(new_audio) > target_length:
        raise ValueError("The trimmed audio is still too long to adjust to length {0}!".format(target_length))
    return PadAudio(new_audio.T, target_length)

def AlignAudioLength(audio1, audio2, mode='pad', trim_scale=1e-5, fixed=None):
    """Align two audio with given mode.
    
    Args:
        audio1: First audio.
        audio2: Second audio.
        mode: Aligning mode which takes its value in
            
            'first', 'second', 'trim', 'mid', 'pad', 'fixed'

            'first': Adjust the length of audio2 to the length of audio1.
            'second':Adjust the length of audio1 to the length of audio2.
            'trim':Trim the two audios first and then adjust them to the length of the shorter one.
            'mid':Trim the two audios first and then pad them to the length of the longger one.
            'pad':Pad two audios into the length of the longger one.
            'fixed':Adjust two audios to the length of a fixed value.
        
        trim_scale: The scale value for TrimAudio.
        fixed: The value for the 'fixed' mode.

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> waveform_hat, sample_rate_hat = torchaudio.load("test_hat.wav")
        >>> len(waveform.T)
            259455
        >>> len(waveform_hat.T)
            259072
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='first')
        >>> len(new_audio1.T)
            259455
        >>> len(new_audio2.T)
            259455
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='second')
            ValueError: The trimmed audio is still too long to adjust to length 259072!
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='trim')
        >>> len(new_audio1.T)
            258519
        >>> len(new_audio2.T)
            258519
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='pad')
        >>> len(new_audio1.T)
            259455
        >>> len(new_audio2.T)
            259455
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='fixed', fixed=307200)
        >>> len(new_audio1.T)
            307200
        >>> len(new_audio2.T)
            307200
    """
    modes = ['first', 'second', 'trim', 'mid', 'pad', 'fixed']
    if not (mode in modes):
        raise("Supported modes: {0}".format(modes))
    
    if mode == 'first':
        new_audio1 = audio1.clone()
        new_audio2 = AdjustAudioLength(audio2, target_length=audio1.shape[1])
    elif mode == 'second':
        new_audio1 = AdjustAudioLength(audio1, target_length=audio2.shape[1])
        new_audio2 = audio1.clone()
    elif mode == 'trim':
        new_audio1 = TrimAudio(audio1, trim_scale=trim_scale)
        new_audio2 = TrimAudio(audio2, trim_scale=trim_scale)
        tmp_length = min(new_audio1.shape[1], new_audio2.shape[1])
        new_audio1 = AdjustAudioLength(new_audio1, target_length=tmp_length, force_trim=True)
        new_audio2 = AdjustAudioLength(new_audio2, target_length=tmp_length, force_trim=True)
    elif mode == 'mid':
        new_audio1 = TrimAudio(audio1, trim_scale=trim_scale)
        new_audio2 = TrimAudio(audio2, trim_scale=trim_scale)
        tmp_length = max(new_audio1.shape[1], new_audio2.shape[1])
        new_audio1 = PadAudio(new_audio1, tmp_length)
        new_audio2 = PadAudio(new_audio2, tmp_length)
    elif mode == 'pad':
        tmp_length = max(audio1.shape[1], audio2.shape[1])
        new_audio1 = PadAudio(audio1, tmp_length)
        new_audio2 = PadAudio(audio2, tmp_length)
    elif mode == 'fixed':
        new_audio1 = AdjustAudioLength(audio1, target_length=fixed, force_trim=True)
        new_audio2 = AdjustAudioLength(audio2, target_length=fixed, force_trim=True)
    return new_audio1, new_audio2

def initialize_midi_settings(
    midi_settings_dir   = get_config('midi_settings_dir'),
    pitch_range         = get_config("default_midi_settings_args")['pitch_range'],
    velocity            = get_config("default_midi_settings_args")['velocity'],
    ticks_per_beat      = get_config("default_midi_settings_args")['ticks_per_beat'],
    duration_beats      = get_config("default_midi_settings_args")['duration_beats'],
    recording_beats     = get_config("default_midi_settings_args")['recording_beats'],
    bpm                 = get_config("default_midi_settings_args")['bpm'],
):
    """Initialize the midi settings."""
    CreateFolder(midi_settings_dir)
    for pitch in range(pitch_range[0], pitch_range[1]):
        midi = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        midi.ticks_per_beat = ticks_per_beat

        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=0))
        track.append(mido.Message('note_off', note=pitch, velocity=velocity, time=duration_beats * ticks_per_beat))
        #The time attribute of each message is the number of seconds since the last message or the start of the file.
        track.append(mido.MetaMessage('end_of_track', time=(recording_beats - duration_beats) * ticks_per_beat))

        midi.save(pjoin(midi_settings_dir, '{0}.mid'.format(pitch)))