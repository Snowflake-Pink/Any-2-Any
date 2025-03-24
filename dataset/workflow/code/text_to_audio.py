# create nodes by instantiation
cliploader_10 = CLIPLoader(clip_name="""t5_base.safetensors""", type="""stable_audio""")
vaedecodeaudio_12 = VAEDecodeAudio()
emptylatentaudio_11 = EmptyLatentAudio(seconds=47.6, batch_size=1)
ksampler_3 = KSampler(seed=558241134136294, control_after_generate="""randomize""", steps=50, cfg=4.98, sampler_name="""dpmpp_3m_sde_gpu""", scheduler="""exponential""", denoise=1)
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""stable-audio-open-1_0.safetensors""")
cliptextencode_6 = CLIPTextEncode(text="""Calm and peaceful nature sounds, including gentle wind rustling through leaves, distant birds chirping, and a light trickling stream. The atmosphere is serene and tranquil, evoking a sense of harmony with nature. Suitable for relaxation or meditation.""")
cliptextencode_7 = CLIPTextEncode(text="""No loud animal noises, no abrupt or sharp sounds, avoid any human-made noises such as traffic, machinery, or voices.""")
saveaudio_13 = SaveAudio(filename_prefix="""audio/ComfyUI""")

# link nodes by invocation
clip_10 = cliploader_10()
latent_11 = emptylatentaudio_11()
model_4, clip_4, vae_4 = checkpointloadersimple_4()
conditioning_7 = cliptextencode_7(clip=clip_10)
conditioning_6 = cliptextencode_6(clip=clip_10)
latent_3 = ksampler_3(model=model_4, positive=conditioning_6, negative=conditioning_7, latent_image=latent_11)
audio_12 = vaedecodeaudio_12(samples=latent_3, vae=vae_4)
result_13 = saveaudio_13(audio=audio_12)
