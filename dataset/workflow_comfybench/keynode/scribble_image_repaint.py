# create nodes by instantiation
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""")
emptylatentimage_5 = EmptyLatentImage(width=512, height=512, batch_size=1)
cliptextencode_6 = CLIPTextEncode(text="""a bird, open wings,""")
cliptextencode_7 = CLIPTextEncode(text="""horror,lowres, zombie,""")
vaedecode_8 = VAEDecode()
