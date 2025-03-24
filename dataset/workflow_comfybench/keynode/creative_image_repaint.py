# create nodes by instantiation
loadimage_1 = LoadImage(image="""letter_r.jpg""")
cliptextencode_2 = CLIPTextEncode(text="""a logo for a game app, bright color""")
cliptextencode_3 = CLIPTextEncode(text="""watermark, blurry, distorted""")
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""majicmixRealistic_v7.safetensors""")
vaeloader_5 = VAELoader(vae_name="""vae-ft-mse-840000-ema-pruned.safetensors""")
vaeencode_6 = VAEEncode()
