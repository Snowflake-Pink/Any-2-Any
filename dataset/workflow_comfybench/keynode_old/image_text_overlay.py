# create nodes by instantiation
emptyimage_16 = EmptyImage(width=512, height=512, batch_size=1, color=0)
crSpaceoverlaySpacetext_18 = CRSpaceOverlaySpaceText(text="""Hello, world!""", font_name="""comic.ttf""", font_size=50, font_color="""custom""", align="""center""", justify="""center""", margins=0, line_spacing=0, position_x=0, position_y=0, rotation_angle=0, rotation_options="""text center""", font_color_hex="""#FFFFFF""")
saveimage_19 = SaveImage(filename_prefix="""ComfyUI""")
