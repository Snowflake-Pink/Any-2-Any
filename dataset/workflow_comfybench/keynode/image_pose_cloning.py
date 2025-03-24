# create nodes by instantiation
ksampler_3 = KSampler(seed=249584040731174, control_after_generate="""randomize""", steps=20, cfg=7, sampler_name="""dpmpp_2m""", scheduler="""karras""", denoise=1)
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""")
emptylatentimage_5 = EmptyLatentImage(width=512, height=512, batch_size=1)
cliptextencode_6 = CLIPTextEncode(text="""a male, dancing in the street""")
cliptextencode_7 = CLIPTextEncode(text="""blurry, painting, drawing, photography, distorted, horror""")
vaedecode_8 = VAEDecode()
