import torch
from model.mulan_pytorch import MuLaN, MuLaNEmbedQuantizer
from model.transformer_audio import AudioSpectrogramTransformer
from model.transformer_text import TextTransformer
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer

# step 1:
audio_transformer = AudioSpectrogramTransformer(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    spec_n_fft=128,
    spec_win_length=24,
    spec_aug_stretch_factor=0.8
)

text_transformer = TextTransformer(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64
)

mulan = MuLaN(
    audio_transformer=audio_transformer,
    text_transformer=text_transformer
)

# get a ton of <sound, text> pairs and train

wavs = torch.randn(2, 1024)
texts = torch.randint(0, 20000, (2, 256))

loss = mulan(wavs, texts)
loss.backward()

# after much training, you can embed sounds and text into a joint embedding space
# for conditioning the audio LM
embeds = mulan.get_audio_latents(wavs)  # during training
embeds = mulan.get_text_latents(texts)  # during inference

# step 2:
quantizer = MuLaNEmbedQuantizer(
    mulan=mulan,  # pass in trained mulan from above
    conditioning_dims=(1024, 1024, 1024),  # say all three transformers have model dimensions of 1024
    namespaces=('semantic', 'coarse', 'fine')
)

# now say you want the conditioning embeddings for semantic transformer
wavs = torch.randn(2, 1024)
conds = quantizer(wavs=wavs, namespace='semantic')  # (2, 8, 1024) - 8 i

# step 3:
wav2vec = HubertWithKmeans(
    checkpoint_path='./hubert/hubert_base_ls960.pt',
    kmeans_path='./hubert/hubert_base_ls960_L9_km500.bin'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens=wav2vec.codebook_size,
    dim=1024,
    depth=6,
    audio_text_condition=True  # this must be set to True (same for CoarseTransformer and FineTransformers)
).cuda()

trainer = SemanticTransformerTrainer(
    transformer=semantic_transformer,
    wav2vec=wav2vec,
    audio_conditioner=quantizer,  # pass in the MulanEmbedQuantizer instance above
    folder='/path/to/audio/files',
    batch_size=1,
    data_max_length=320 * 32,
    num_train_steps=1
)

trainer.train()

