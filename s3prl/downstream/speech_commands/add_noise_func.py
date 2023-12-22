import random
from .Add_Bg_Noise import AddBackgroundNoise

def add_noise(clean_audio):
    top_value = 12
    buttom_value = 1
    id = random.randint(buttom_value, top_value)
    nosiy_path = '/temp/add_noise/noise_lib/Babble_' + str(id) + '.wav'
    augment = AddBackgroundNoise(
        sounds_path=nosiy_path,
        p=1
    )
    audio_with_noise, snr = augment(clean_audio, 16000)
    return audio_with_noise
