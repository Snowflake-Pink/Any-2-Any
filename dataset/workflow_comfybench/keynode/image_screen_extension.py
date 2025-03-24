# create nodes by instantiation
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""")
cliptextencode_6 = CLIPTextEncode(text="""an image of iceberg""")
cliptextencode_7 = CLIPTextEncode(text="""illustration, painting, text, watermark, copyright, signature, notes""")
imagepadforoutpaint_11 = ImagePadForOutpaint(left=256, top=0, right=256, bottom=0, feathering=0)
vaeencodeforinpaint_12 = VAEEncodeForInpaint(grow_mask_by=16)
