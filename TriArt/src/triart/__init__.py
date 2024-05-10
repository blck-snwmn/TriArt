from triart.polygonizer import (
    polygonize_image,
    save_image,
    enhanced_polygonize_image,
    enhanced_polygonize_image2,
    enhanced_polygonize_image3,
)


def main() -> int:
    image_path = "shikunIMG_9107_TP_V4.jpg"  # 画像パスを適切に設定
    # result_image = polygonize_image(image_path)
    # result_image = enhanced_polygonize_image(image_path)
    # result_image = enhanced_polygonize_image2(image_path)
    result_image = enhanced_polygonize_image3(image_path)
    save_path = "path_to_save_result.jpg"  # 保存パスを適切に設定
    save_image(result_image, save_path)
    return 0
