# create nodes by instantiation
vaedecode_8 = VAEDecode()
ksampler_3 = KSampler(seed=152125378669911, control_after_generate="""randomize""", steps=20, cfg=8, sampler_name="""uni_pc_bh2""", scheduler="""normal""", denoise=1)
checkpointloadersimple_29 = CheckpointLoaderSimple(ckpt_name="""512-inpainting-ema.safetensors""")
imagepadforoutpaint_30 = ImagePadForOutpaint(left=0, top=128, right=0, bottom=128, feathering=40)
saveimage_9 = SaveImage(filename_prefix="""ComfyUI""")
vaeencodeforinpaint_26 = VAEEncodeForInpaint(grow_mask_by=8)
