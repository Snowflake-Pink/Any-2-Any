# create nodes by instantiation
saveimage_9 = SaveImage(filename_prefix="""ComfyUI""")
ksampler_3 = KSampler(seed=636250194499614, control_after_generate="""randomize""", steps=20, cfg=7, sampler_name="""dpmpp_2m""", scheduler="""karras""", denoise=1)
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""")
vaedecode_1 = VAEDecode()