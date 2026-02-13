def resize_image(img_path: str, width: int, height: int) -> str:
    """
    Resize an image to specific dimensions.
    """
    return f"Resized image at {img_path} to {width}x{height}"

def convert_format(source_file: str, target_format: str) -> str:
    """
    Convert image format (e.g., jpg to png).
    """
    return f"Converted {source_file} to {target_format}"
