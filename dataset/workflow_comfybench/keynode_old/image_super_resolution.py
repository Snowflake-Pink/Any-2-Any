# create nodes by instantiation
upscalemodelloader_11 = UpscaleModelLoader(model_name="""4x-UltraSharp.pth""")
imageupscalewithmodel_12 = ImageUpscaleWithModel()
saveimage_29 = SaveImage(filename_prefix="""ComfyUI""")
imagescaleby_30 = ImageScaleBy(upscale_method="""bilinear""", scale_by=0.5)
loadimage_31 = LoadImage(image="""titled_book.png""")
