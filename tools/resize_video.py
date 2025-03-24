import cv2

def resize_video(input_path, output_path, target_length=256):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError()

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"origin size: {original_width}x{original_height}")


    if original_width >= original_height:
        new_width = target_length
        new_height = int(original_height * (target_length / original_width))
    else:
        new_height = target_length
        new_width = int(original_width * (target_length / original_height))

    new_width = new_width // 2 * 2
    new_height = new_height // 2 * 2
    print(f"size: {new_width}x{new_height}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (new_width, new_height))
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"save to：{output_path}")

if __name__ == "__main__":

    resize_video(
        input_path="./resources/taylor_swift.mp4",
        output_path="ComfyUI-0.1.3/taylor_swift.mp4",
        target_length=256
    )