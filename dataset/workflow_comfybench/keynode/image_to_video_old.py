# create nodes by instantiation
loadimage_50 = LoadImage(image="""play_guitar.jpg""")
svd_img2vid_conditioning_63 = SVD_img2vid_Conditioning(width=1024, height=576, video_frames=24, motion_bucket_id=100, fps=6, augmentation_level=0)
videolinearcfgguidance_89 = VideoLinearCFGGuidance(min_cfg=1)
