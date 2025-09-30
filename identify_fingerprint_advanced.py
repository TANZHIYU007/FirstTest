import cv2, os, re

SRC = "data/src.bmp"
DB_DIR = "data/database"

# 识别阶段：ratio 宽松一些
RATIO = 0.80
NOT_FOUND_THRESHOLD = 15   # 最高匹配数低于它则判定未识别
TOPK = 3

# 预览缩放倍数（你的图较小，放大看更清晰）
PREVIEW_SCALE = 3
OUTPUT_PATH = "output/best_match.png"


def knn_good_pairs(des1, des2, ratio=RATIO):
    """返回三样:
       - goods: 通过 ratio test 的好匹配（DMatch 列表）
       - pairs_for_draw: 画线用的 [[m, n], ...]
       - knn_matches: 原始 knn 匹配结果
    """
    flann = cv2.FlannBasedMatcher()
    knn_matches = flann.knnMatch(des1, des2, k=2)
    goods, pairs_for_draw = [], []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            goods.append(m)
            pairs_for_draw.append([m, n])  # drawMatchesKnn 需要 list[list[DMatch]]
    return goods, pairs_for_draw, knn_matches


def extract_id_from_name(fname: str) -> int:
    m = re.match(r"(\d+)", os.path.basename(fname))
    return int(m.group(1)) if m else 9999


def resize_for_preview(img, scale=PREVIEW_SCALE):
    if scale == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def main():
    # 读取 src
    src = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("❌ 无法读取", SRC)
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

    # 遍历数据库并记录结果（含可视化所需数据）
    if not os.path.isdir(DB_DIR):
        print("❌ 找不到数据库目录:", DB_DIR)
        return

    results = []  # (fname, good_count, kp_db, des_db, img_db, pairs_for_draw)
    for fname in os.listdir(DB_DIR):
        fpath = os.path.join(DB_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        img_db = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img_db is None:
            continue
        kp_db, des_db = sift.detectAndCompute(img_db, None)
        if des_db is None:
            results.append((fname, 0, kp_db, des_db, img_db, []))
            continue
        goods, pairs_for_draw, _ = knn_good_pairs(des_src, des_db, RATIO)
        results.append((fname, len(goods), kp_db, des_db, img_db, pairs_for_draw))

    if not results:
        print("❌ 数据库为空")
        return

    # 排序 & 打印前几名
    results.sort(key=lambda x: x[1], reverse=True)
    print("▶ 匹配结果（前几名）:")
    for fname, n, *_ in results[:TOPK]:
        print(f"  {fname:<25}  good={n}")

    # 取最佳
    best_name, best_score, kp_best, des_best, img_best, pairs_best = results[0]
    best_id = extract_id_from_name(best_name)

    if best_score < NOT_FOUND_THRESHOLD:
        print(f"\n🔎 识别结果：未识别（最高匹配数={best_score} < 阈值 {NOT_FOUND_THRESHOLD}）")
    else:
        print(f"\n✅ 识别结果：ID={best_id}  文件={best_name}  (最高匹配数={best_score})")

    # ===== 可视化 =====
    # 1) 并排显示 best 匹配（带连线）
    matched = cv2.drawMatchesKnn(
        src, kp_src,
        img_best, kp_best,
        pairs_best, None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    # 放大预览更清晰
    matched_preview = resize_for_preview(matched, PREVIEW_SCALE)
    src_preview = resize_for_preview(src, PREVIEW_SCALE)
    best_preview = resize_for_preview(img_best, PREVIEW_SCALE)

    cv2.imshow("src (input)", src_preview)
    cv2.imshow(f"best match: {best_name}", best_preview)
    cv2.imshow("Matches (press any key to close)", matched_preview)

    # 保存到文件
    ensure_dir(OUTPUT_PATH)
    cv2.imwrite(OUTPUT_PATH, matched)
    print(f"\n🖼️ 已保存可视化结果到: {OUTPUT_PATH}")

    cv2.waitKey(0)   # 按任意键关闭窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
