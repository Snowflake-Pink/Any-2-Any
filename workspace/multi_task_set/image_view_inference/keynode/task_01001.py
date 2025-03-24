# create nodes by instantiation
ksampler_3 = KSampler(seed=237514639057560, control_after_generate="""fixed""", steps=20, cfg=5, sampler_name="""euler""", scheduler="""sgm_uniform""", denoise=1)
stablezero123_conditioning_26 = StableZero123_Conditioning(width=256, height=256, batch_size=1, elevation=10, azimuth=142)
imageonlycheckpointloader_15 = ImageOnlyCheckpointLoader(ckpt_name="""stable_zero123.ckpt""")
