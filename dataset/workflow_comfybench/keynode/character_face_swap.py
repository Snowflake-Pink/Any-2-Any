# create nodes by instantiation
reactorfaceswap_13 = ReActorFaceSwap(enabled=True, swap_model="""inswapper_128.onnx""", facedetection="""retinaface_resnet50""", face_restore_model="""codeformer-v0.1.0.pth""", face_restore_visibility=1, codeformer_weight=0.5, detect_gender_input="""no""", detect_gender_source="""no""", input_faces_index="""0""", source_faces_index="""0""", console_log_level=0)
loadimage_18 = LoadImage(image="""source.jpg""")
saveimage_20 = SaveImage(filename_prefix="""swapped""")
