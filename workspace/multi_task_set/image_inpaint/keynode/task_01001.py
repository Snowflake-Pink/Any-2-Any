# create nodes by instantiation
vaeencodeforinpaint_26 = VAEEncodeForInpaint(grow_mask_by=6)
checkpointloadersimple_29 = CheckpointLoaderSimple(ckpt_name="""512-inpainting-ema.safetensors""")
ksampler_3 = KSampler(seed=406402661964333, control_after_generate="""randomize""", steps=20, cfg=8, sampler_name="""uni_pc_bh2""", scheduler="""normal""", denoise=1)
saveimage_9 = SaveImage(filename_prefix="""ComfyUI""")
