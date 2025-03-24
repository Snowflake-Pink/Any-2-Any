# create nodes by instantiation
cliploader_10 = CLIPLoader(clip_name="""t5_base.safetensors""", type="""stable_audio""")
vaedecodeaudio_12 = VAEDecodeAudio()
emptylatentaudio_11 = EmptyLatentAudio(seconds=10, batch_size=1)
saveaudio_13 = SaveAudio(filename_prefix="""audio/ComfyUI""")
ksampler_3 = KSampler(seed=1022482648933445, control_after_generate="""randomize""", steps=50, cfg=4.98, sampler_name="""dpmpp_3m_sde_gpu""", scheduler="""exponential""", denoise=1)
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""stable-audio-open-1_0.safetensors""")
