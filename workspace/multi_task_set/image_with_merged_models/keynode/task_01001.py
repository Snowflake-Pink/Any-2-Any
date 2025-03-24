# create nodes by instantiation
ksampler_3 = KSampler(seed=156680208700286, control_after_generate="""fixed""", steps=20, cfg=8, sampler_name="""euler""", scheduler="""normal""", denoise=1)
emptylatentimage_5 = EmptyLatentImage(width=512, height=512, batch_size=1)
vaeloader_15 = VAELoader(vae_name="""kl-f8-anime2.ckpt""")
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""v1-5-pruned-emaonly.ckpt""")
checkpointloadersimple_11 = CheckpointLoaderSimple(ckpt_name="""cardosAnime_v10.safetensors""")
checkpointsave_14 = CheckpointSave(filename_prefix="""checkpoints/ComfyUI""")
modelmergesimple_17 = ModelMergeSimple(ratio=0.49999999999999956)
