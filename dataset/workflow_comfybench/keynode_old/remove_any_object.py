# create nodes by instantiation
loadimage_33 = LoadImage(image="""bedroom.jpg""")
lamainpaint_84 = LaMaInpaint(device_mode="""Prefer GPU""")
groundingdinosamsegmentSpaceLeftParensegmentSpaceanythingRightParen_101 = GroundingDinoSAMSegmentSpaceLeftParensegmentSpaceanythingRightParen(prompt="""chair""", threshold=0.2)
groundingdinomodelloaderSpaceLeftParensegmentSpaceanythingRightParen_102 = GroundingDinoModelLoaderSpaceLeftParensegmentSpaceanythingRightParen(model_name="""GroundingDINO_SwinT_OGC (694MB)""")
sammodelloaderSpaceLeftParensegmentSpaceanythingRightParen_103 = SAMModelLoaderSpaceLeftParensegmentSpaceanythingRightParen(model_name="""sam_vit_l (1.25GB)""")
saveimage_113 = SaveImage(filename_prefix="""ComfyUI""")
growmask_114 = GrowMask(expand=5, tapered_corners=True)
