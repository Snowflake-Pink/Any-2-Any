# create nodes by instantiation
checkpointloadersimple_1 = CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""")
vaeloader_2 = VAELoader(vae_name="""vae-ft-mse-840000-ema-pruned.safetensors""")
ipadaptermodelloader_3 = IPAdapterModelLoader(ipadapter_file="""ip-adapter_sd15.safetensors""")
clipvisionloader_4 = CLIPVisionLoader(clip_name="""CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors""")
loadimage_6 = LoadImage(image="""woman_portrait.jpg""")
cliptextencode_7 = CLIPTextEncode(text="""beautiful renaissance girl, detailed""")
