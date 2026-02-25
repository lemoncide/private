def resize_image(img_path: str, width: int, height: int) -> str:
    """
    调整图片尺寸（示例实现）。

    何时用：需要把图片缩放到指定宽高后再进行展示/上传/进一步处理。
    输入：img_path（图片路径）、width/height（目标尺寸，像素）。
    输出：处理结果说明（str）。
    典型任务：图片预处理、生成缩略图、统一素材尺寸。
    """
    return f"Resized image at {img_path} to {width}x{height}"

def convert_format(source_file: str, target_format: str) -> str:
    """
    转换图片格式（示例实现）。

    何时用：需要把图片从一种格式转换为另一种格式（如 jpg→png）。
    输入：source_file（源文件路径）、target_format（目标格式，如 png/jpg/webp）。
    输出：处理结果说明（str）。
    典型任务：格式统一、兼容性处理、压缩链路前的格式转换。
    """
    return f"Converted {source_file} to {target_format}"
