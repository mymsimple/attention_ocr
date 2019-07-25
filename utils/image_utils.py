import cv2

# 图像缩放，高度都是32
def resize_batch_image(image_names: list):

    resized_images = []

    for image_name, label_id in image_names:
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        h,w,_ = image.shape
        ratio = h/32
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        resized_images.append(image)

    return resized_images
