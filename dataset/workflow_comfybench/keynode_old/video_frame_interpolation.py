# create nodes by instantiation
vhs_loadvideo_7 = VHS_LoadVideo(video="""male_idol.mp4""", force_rate=0, force_size="""Disabled""", custom_width=512, custom_height=512, frame_load_cap=0, skip_first_frames=0, select_every_nth=1)
rifeSpacevfi_10 = RIFESpaceVFI(ckpt_name="""rife47.pth""", clear_cache_after_n_frames=10, multiplier=3, fast_mode=True, ensemble=True, scale_factor=1)
saveanimatedwebp_11 = SaveAnimatedWEBP(filename_prefix="""ComfyUI""", fps=24, lossless=True, quality=80, method="""default""")
