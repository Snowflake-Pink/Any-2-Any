# create nodes by instantiation
ksampler_3 = KSampler(seed=381979438476462, control_after_generate="""randomize""", steps=18, cfg=6, sampler_name="""ddpm""", scheduler="""karras""", denoise=1)
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""majicmixRealistic_v7.safetensors""")
emptylatentimage_5 = EmptyLatentImage(width=768, height=512, batch_size=1)
cliptextencode_7 = CLIPTextEncode(text="""blurry, illustration, distorted, cropped""")
vaedecode_8 = VAEDecode()
vaeloader_11 = VAELoader(vae_name="""vae-ft-mse-840000-ema-pruned.safetensors""")
cliptextencode_17 = CLIPTextEncode(text="""Godzilla raising from the water, near a Caribbean beach  high resolution, high quality, detailed, 4k""")
