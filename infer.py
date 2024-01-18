from model.musiclm_pytorch import MusicLM
import torchaudio
from einops import rearrange

prompt = 'the crystalline sounds of the piano in a ballroom'
save_path = "/path/to/save/music"

# Refer to the train folder
musiclm = MusicLM(audio_lm=audio_lm, mulan_embed_quantizer=quantizer)
music = musiclm.forward(prompt, num_samples=4)
generated_wave = rearrange(music, 'b n -> b 1 n')
torchaudio.save(save_path, generated_wave, musiclm.neural_codec.sample_rate)
