import cv2

# 图片路径
SRC = "data/src.bmp"
MODEL = "data/model.bmp"

# Lowe's ratio 测试阈值（教程里用 0.65）
RATIO = 0.65

# 匹配数量阈值（教程里用 500 作为示例）
PASS_THRESHOLD = 32


def main():
    # 读取灰度图
    src = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE)
    model = cv2.imread(MODEL, cv2.IMREAD_GRAYSCALE)

    if src is None or model is None:
        print("❌ 无法读取图像，请确认路径和格式")
        return

    # 创建 SIFT 特征提取器
    sift = cv2.SIFT_create()

    # 提取关键点和描述符
    kp1, des1 = sift.detectAndCompute(src, None)
    kp2, des2 = sift.detectAndCompute(model, None)

    if des1 is None or des2 is None:
        print("❌ 没有检测到足够的特征点")
        return

    # 使用 FLANN 进行匹配
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < RATIO * n.distance]

    print(f"✅ 匹配到的有效特征点数: {len(good)}")

    if len(good) >= PASS_THRESHOLD:
        print("🔓 Authentication Successful （验证通过）")
    else:
        print("🔒 Authentication Failed （验证失败）")


if __name__ == "__main__":
    main()
