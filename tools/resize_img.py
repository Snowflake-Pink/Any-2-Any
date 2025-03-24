from PIL import Image

def resize_image(input_path, output_path, size):
    """
    Resize the image to the specified size and save it.

    :param input_path: Path to the input image
    :param output_path: Path to save the resized image
    :param size: Tuple (width, height) specifying the new size
    """
    with Image.open(input_path) as img:
        resized_img = img.resize(size)
        resized_img.save(output_path)

if __name__ == "__main__":
    resize_image('./resources/underwater_city.png', '/home/ComfyUI-0.1.3/input/underwater_city.png', (1024, 512))
    resize_image('./resources/underwater_city.png', './resources/underwater_city.png', (1024, 512))