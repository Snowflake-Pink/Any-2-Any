# create nodes by instantiation
loadimage_50 = LoadImage(image="""play_guitar.jpg""")
svd_img2vid_conditioning_63 = SVD_img2vid_Conditioning(width=1024, height=576, video_frames=24, motion_bucket_id=100, fps=6, augmentation_level=0)
videolinearcfgguidance_89 = VideoLinearCFGGuidance(min_cfg=1)
ksampleradvanced_92 = KSamplerAdvanced(add_noise="""enable""", noise_seed=942806259821634, control_after_generate="""randomize""", steps=20, cfg=2.52, sampler_name="""euler""", scheduler="""ddim_uniform""", start_at_step=0, end_at_step=10000, return_with_leftover_noise="""disable""")
imageonlycheckpointloader_64 = ImageOnlyCheckpointLoader(ckpt_name="""svd_xt_1_1.safetensors""")
vaedecode_70 = VAEDecode()
saveanimatedwebp_96 = SaveAnimatedWEBP(filename_prefix="""ComfyUI""", fps=8, lossless=True, quality=80, method="""default""")
