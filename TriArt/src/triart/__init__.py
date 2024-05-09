from triart.polygonizer import polygonize_image, save_image, enhanced_polygonize_image


def main() -> int:
    image_path = "16776570719onz6-300x300.jpg"  # 画像パスを適切に設定
    result_image = enhanced_polygonize_image(image_path)
    save_path = "path_to_save_result.jpg"  # 保存パスを適切に設定
    save_image(result_image, save_path)
    return 0
