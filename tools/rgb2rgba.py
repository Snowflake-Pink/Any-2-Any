from PIL import Image
import os
import json

def convert_to_rgba(image_path, output_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert the image to RGBA format
        rgba_image = img.convert("RGBA")
        # Save the converted image
        rgba_image.save(output_path)
        print(f"Image converted to RGBA and saved to {output_path}")

if __name__=="__main__":
    meta_path = ""
    input_dir = ""
    convert_to_rgba(meta_path, input_dir)
    # meta_path = ""
    # input_dir = ""
    # with open(meta_path,'r') as f:
    #     meta_json = json.load(f)
    # for task_name, value in meta_json.items():
    #     img = value['content'].split("'")[1]
    #     # print(img)
    #     if img.endswith("png") or img.endswith("jpg") or img.endswith("jpeg") and 'rgba' not in img:
    #         convert_to_rgba(os.path.join(input_dir, img), os.path.join(input_dir, img.split(".")[0] + "_rgba.png"))
