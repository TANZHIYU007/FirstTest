import cv2, os

# 要检查的关键文件路径
paths = ["data/src.bmp", "data/model.bmp"]

for p in paths:
    # 打印文件是否存在
    print(f"[check] {p}: ", "EXISTS" if os.path.exists(p) else "MISSING")

    # 尝试用 OpenCV 读取图片（灰度模式）
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    # 如果能读出来，说明路径正确，格式没问题
    print("  read:", "OK" if img is not None else "FAIL",
          "| shape:", None if img is None else img.shape)

# 检查 database 文件夹
db = "data/database"
if not os.path.isdir(db):
    # 如果不存在，提示错误
    print("[check] data/database: MISSING")
else:
    # 列出文件夹里的文件数量和前几个文件名
    files = [f for f in os.listdir(db) if os.path.isfile(os.path.join(db, f))]
    print(f"[check] data/database: {len(files)} files ->",
          files[:5], "..." if len(files) > 5 else "")
