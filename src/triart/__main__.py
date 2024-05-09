from triart.polygonizer import polygonize_image, save_image

def main():
    image_path = 'path_to_your_image.jpg'  # 画像パスを適切に設定
    result_image = polygonize_image(image_path)
    save_path = 'path_to_save_result.jpg'  # 保存パスを適切に設定
    save_image(result_image, save_path)

if __name__ == "__main__":
    main()
