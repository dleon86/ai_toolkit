from PIL import Image, ImageFilter

def resize_image(image_path: str, output_path: str, size: tuple) -> None:
    """
    Resizes the image to the specified size.

    Parameters:
    - image_path (str): Path to the input image.
    - output_path (str): Path to save the resized image.
    - size (tuple): New size as (width, height).
    """
    with Image.open(image_path) as img:
        resized_img = img.resize(size)
        resized_img.save(output_path)
        print(f"Image resized to {size} and saved to {output_path}")

def apply_filter(image_path: str, output_path: str, filter_type: str) -> None:
    """
    Applies a filter to the image.

    Parameters:
    - image_path (str): Path to the input image.
    - output_path (str): Path to save the filtered image.
    - filter_type (str): Type of filter to apply (e.g., BLUR, CONTOUR).
    """
    filter_mapping = {
        "BLUR": ImageFilter.BLUR,
        "CONTOUR": ImageFilter.CONTOUR,
        "DETAIL": ImageFilter.DETAIL,
        "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE
    }

    with Image.open(image_path) as img:
        if filter_type in filter_mapping:
            filtered_img = img.filter(filter_mapping[filter_type])
            filtered_img.save(output_path)
            print(f"Applied {filter_type} filter and saved to {output_path}")
        else:
            print(f"Filter type {filter_type} is not supported.") 