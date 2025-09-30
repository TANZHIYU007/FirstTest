import cv2, os, re

SRC = "data/src.bmp"
DB_DIR = "data/database"

# 根据教程思路：识别用更宽松的 ratio
RATIO = 0.80

# 你的图分辨率较小（~103×96），匹配点整体会偏低
# 建议先用 15 作为“未识别”阈值；后续可按你的数据再调整
NOT_FOUND_THRESHOLD = 15

# 显示前几名候选
TOPK = 3

def good_matches(des1, des2, ratio=RATIO):
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < ratio * n.distance]

def extract_id_from_name(fname: str) -> int:
    """
    教程做法：用文件名前缀的数字做 ID（如 0_xxx.bmp -> 0）
    我们更健壮：取开头连续数字；没有则返回 9999（未知）
    """
    m = re.match(r"(\d+)", os.path.basename(fname))
    return int(m.group(1)) if m else 9999

def main():
    # 读 src
    src = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("❌ 无法读取 data/src.bmp")
        return

    # SIFT
    if not hasattr(cv2, "SIFT_create"):
        print("❌ 当前 OpenCV 不含 SIFT，请安装 opencv-contrib-python")
        return
    sift = cv2.SIFT_create()
    kp_src, des_src = sift.detectAndCompute(src, None)
    if des_src is None:
        print("❌ src 未检测到足够特征点")
        return

    # 遍历数据库
    if not os.path.isdir(DB_DIR):
        print("❌ 找不到数据库目录:", DB_DIR)
        return

    results = []  # (fname, num_good)
    for fname in os.listdir(DB_DIR):
        fpath = os.path.join(DB_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp_db, des_db = sift.detectAndCompute(img, None)
        if des_db is None:
            results.append((fname, 0))
            continue
        good = good_matches(des_src, des_db)
        results.append((fname, len(good)))

    if not results:
        print("❌ 数据库为空")
        return

    # 按匹配数排序、打印前几名
    results.sort(key=lambda x: x[1], reverse=True)
    print("▶ 匹配结果（前几名）:")
    for fname, n in results[:TOPK]:
        print(f"  {fname:<25}  good={n}")

    best_name, best_score = results[0]
    best_id = extract_id_from_name(best_name)

    if best_score < NOT_FOUND_THRESHOLD:
        print(f"\n🔎 识别结果：未识别（最高匹配数={best_score} < 阈值 {NOT_FOUND_THRESHOLD}）")
    else:
        print(f"\n✅ 识别结果：ID={best_id}  文件={best_name}  (最高匹配数={best_score})")

if __name__ == "__main__":
    main()
