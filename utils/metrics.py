from .audio_utils import *

class AudioMetric:
    """Create a global frame for calculating audio metrics."""
    def __init__(self):
        self.AUDIO = {}
        self.PARAM = {}
        self.SPECT = {}
        self.OTHER = {}
    
    def register(self, mode):
        if mode == 'audio':
            def r(f):
                self.AUDIO[f.__name__] = f
                return f
            return r
        if mode == 'param':
            def r(f):
                self.PARAM[f.__name__] = f
                return f
            return r
        if mode == 'spect':
            def r(f):
                self.SPECT[f.__name__] = f
                return f
            return r
        if mode == 'other':
            def r(f):
                self.OTHER[f.__name__] = f
                return f
            return r

METRIC = AudioMetric()

def _args_to_metric_args(**args):
    """Adjust the args."""
    metric_args = deepcopy(args)
    if '_use_mfccd' in metric_args:
        if metric_args.pop('_use_mfccd'):
            metric_args['n_mfcc'] = get_config("default_metric_args")['n_mfcc']
    
    if '_use_mssmae' in metric_args:
        if 'sample_rate' in metric_args:
            metric_args.pop('sample_rate')
        i = metric_args.pop('_use_mssmae')
        metric_args['n_fft'] = get_config("default_metric_args")['n_ffts'][i]
        metric_args['hop_length'] = get_config("default_metric_args")['n_hops'][i]
    return metric_args

@METRIC.register(mode='spect')
def MSE(matrix1, matrix2):
    return ((matrix1 - matrix2) ** 2).mean()

@METRIC.register(mode='spect')
def MAE(matrix1, matrix2):
    return ((matrix1 - matrix2).abs()).mean()

@METRIC.register(mode='spect')
def LSD(matrix1, matrix2):
    return ((matrix1 - matrix2) ** 2).mean(dim=-1).sqrt().mean()

@METRIC.register(mode='other')
def COV(vector1, vector2):
    return (vector1 * vector2).mean() - vector1.mean() * vector2.mean()

@METRIC.register(mode='other')
def PCC(vector1, vector2):
    return COV(vector1, vector2) / vector1.std() / vector2.std()

# Attention: 'sample_rate' should be specified
@METRIC.register(mode='audio')
def AudioMSE(audio1, audio2, **melargs):
    return MSE(AudioToMel(audio1, **melargs), AudioToMel(audio2, **melargs))

# Attention: 'sample_rate' should be specified
@METRIC.register(mode='audio')
def AudioMAE(audio1, audio2, **melargs):
    return MAE(AudioToMel(audio1, **melargs), AudioToMel(audio2, **melargs))

# Attention: 'sample_rate' should be specified
@METRIC.register(mode='audio')
def AudioLSD(audio1, audio2, **melargs):
    return LSD(AudioToMel(audio1, **melargs), AudioToMel(audio2, **melargs))

# Attention: 'sample_rate' should be specified
@METRIC.register(mode='audio')
def AudioMFCCD(audio1, audio2, **mfccargs):
    return LSD(AudioToMFCC(audio1, **_args_to_metric_args(_use_mfccd=True, **mfccargs)), AudioToMFCC(audio2, **_args_to_metric_args(_use_mfccd=True, **mfccargs)))

# Attention: 'sample_rate' should be specified
@METRIC.register(mode='audio')
def AudioPCC(audio1, audio2, **melargs):
    mel1 = AudioToMel(audio1, **melargs)
    mel2 = AudioToMel(audio2, **melargs)
    #The output of AudioToMel is in the shape of (channel, n_mels, time).
    mel1 = mel1.reshape(mel1.shape[0], -1)
    mel2 = mel2.reshape(mel2.shape[0], -1)
    return PCC(mel1, mel2)

# Attention: 'sample_rate' should be specified
@METRIC.register(mode='audio')
def AudioMSSMAE(audio1, audio2, log_margin=1e-10, **specargs):
    loss = 0
    fft_sizes = len(get_config("default_metric_args")['n_ffts'])
    a = get_config("default_metric_args")['alpha']
    for i in range(fft_sizes):
        spec1 = AudioToSpec(audio1, **_args_to_metric_args(_use_mssmae=i, **specargs))
        spec2 = AudioToSpec(audio2, **_args_to_metric_args(_use_mssmae=i, **specargs))
        loss += MAE(spec1, spec2) + a * MAE(torch.log(spec1+log_margin), torch.log(spec2+log_margin))
    return loss / fft_sizes

def ComputeMetrics(pd_file, gt_file):
    """Compute the metrics.
    
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> waveform_hat, sample_rate_hat = torchaudio.load("test_hat.wav")
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='pad')
        >>> path1 = "audio1.wav"
        >>> path2 = "audio2.wav"
        >>> torchaudio.save(path1, new_audio1, sample_rate)
        >>> torchaudio.save(path2, new_audio2, sample_rate)
        >>> ComputeMetrics(path1, path2)
            {'count': 1, 
            'AudioMSE': 0.0021917209960520267, 
            'AudioMAE': 0.0052396394312381744, 
            'AudioLSD': 0.029458042234182358, 
            'AudioMFCCD': 0.9125940203666687, 
            'AudioPCC': 0.9995368719100952, 
            'AudioMSSMAE': 0.14594422280788422, 
            'MSE': 0.9229055643081665, 
            'MAE': 0.06717035919427872, 
            'LSD': 0.40060627460479736, 
            'COV': 382.4127197265625, 
            'PCC': 0.9999992251396179}
    """
    profile = {'count':1}
    pd, sr = tau.load(pd_file)
    gt, sr = tau.load(gt_file)
    pd_spec = AudioToSpec(pd)
    gt_spec = AudioToSpec(gt)
    for metric in METRIC.AUDIO:
        profile[metric] = float(METRIC.AUDIO[metric](pd, gt))
    for metric in METRIC.SPECT:
        profile[metric] = float(METRIC.SPECT[metric](pd_spec, gt_spec))
    for metric in METRIC.OTHER:
        profile[metric] = float(METRIC.OTHER[metric](pd_spec, pd_spec))
    return profile

def Evaluate(folder, tqdm=True):
    """Evaluate the losses between Pred files and GroundTruth files in the given folder.
        
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav")
        >>> waveform_hat, sample_rate_hat = torchaudio.load("test_hat.wav")
        >>> new_audio1, new_audio2 = AlignAudioLength(waveform, waveform_hat, mode='pad')
        >>> path1 = "audio1_gt.wav"
        >>> path2 = "audio1_pd.wav"
        >>> torchaudio.save(path1, new_audio1, sample_rate)
        >>> torchaudio.save(path2, new_audio2, sample_rate)
        >>> path = "/home/yichenggu/Sound2Synth-Linux-Reproduction"
        >>> Evaluate(path)
            {'count': 1, 
            'AudioMSE': 0.0021917209960520267, 
            'AudioMAE': 0.0052396394312381744, 
            'AudioLSD': 0.029458042234182358, 
            'AudioMFCCD': 0.9125940203666687, 
            'AudioPCC': 0.99953693151474, 
            'AudioMSSMAE': 0.14594422280788422, 
            'MSE': 0.9229055643081665, 
            'MAE': 0.06717035919427872, 
            'LSD': 0.40060627460479736, 
            'COV': 381.51593017578125, 
            'PCC': 0.9999992251396179}
    """
    profile = {}
    instances = list(set(file.split('_')[0] for file in ListFiles(folder)))
    #TQDM is an iterable pbar defined in pyheaven.
    for instance in (TQDM(instances) if tqdm else instances):
        pd_file = pjoin(folder, instance+"_pd.wav")
        gt_file = pjoin(folder, instance+"_gt.wav")
        if ExistFile(pd_file) and ExistFile(gt_file):
            instance_profile = ComputeMetrics(pd_file, gt_file)
            for key in instance_profile:
                profile[key] = profile[key]+instance_profile[key] if key in profile else instance_profile[key]
    return profile