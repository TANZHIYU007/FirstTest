# fingerprint_core.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os, re, cv2, numpy as np

@dataclass
class MatchCandidate:
    filename: str
    good_matches: int

@dataclass
class IdentifyResult:
    ok: bool                     # 是否识别成功（最高分 >= 阈值）
    best_id: Optional[int]       # 从文件名前缀抽取的 ID，失败则 None
    best_filename: Optional[str] # 最佳匹配的文件名
    best_score: int              # 最佳匹配的 good 数
    topk: List[MatchCandidate]   # 前几名候选（按分数降序）
    vis_path: Optional[str]      # 可视化保存路径（可能为 None）
    message: str                 # 结果消息（友好提示）

def _extract_id_from_name(fname: str) -> Optional[int]:
    m = re.match(r"(\d+)", os.path.basename(fname))
    return int(m.group(1)) if m else None

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _resize_preview(img: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

def identify_in_db(
    src_path: str,
    db_dir: str,
    ratio: float = 0.80,
    not_found_threshold: int = 15,
    topk_n: int = 3,
    save_vis: bool = True,
    vis_output_path: str = "output/best_match.png",
    preview_scale: int = 3
) -> IdentifyResult:
    """
    用 SIFT + FLANN + Lowe's ratio test 在数据库里识别 src 图片。
    返回 IdentifyResult，包含是否识别成功、最佳匹配、候选列表和可视化文件。
    """
    if not os.path.exists(src_path):
        return IdentifyResult(False, None, None, 0, [], None, f"找不到输入图像：{src_path}")
    if not os.path.isdir(db_dir):
        return IdentifyResult(False, None, None, 0, [], None, f"找不到数据库目录：{db_dir}")

    # 读 src（灰度）
    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        return IdentifyResult(False, None, None, 0, [], None, "无法读取输入图像（格式或路径有误）")

    # SIFT
    if not hasattr(cv2, "SIFT_create"):
        return IdentifyResult(False, None, None, 0, [], None, "当前 OpenCV 不含 SIFT，请安装 opencv-contrib-python")
    sift = cv2.SIFT_create()
    kp_src, des_src = sift.detectAndCompute(src, None)
    if des_src is None:
        return IdentifyResult(False, None, None, 0, [], None, "输入图像未检测到足够特征点")

    # 遍历数据库
    flann = cv2.FlannBasedMatcher()
    results: List[Tuple[str, int, list, list, np.ndarray, list]] = []
    # ↑ (fname, score, kp_db, des_db, img_db, pairs_for_draw)

    for fname in os.listdir(db_dir):
        fpath = os.path.join(db_dir, fname)
        if not os.path.isfile(fpath):
            continue
        img_db = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img_db is None:
            continue
        kp_db, des_db = sift.detectAndCompute(img_db, None)
        if des_db is None:
            results.append((fname, 0, kp_db, des_db, img_db, []))
            continue
        knn = flann.knnMatch(des_src, des_db, k=2)
        goods, pairs = [], []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                goods.append(m)
                pairs.append([m, n])
        results.append((fname, len(goods), kp_db, des_db, img_db, pairs))

    if not results:
        return IdentifyResult(False, None, None, 0, [], None, "数据库为空或无法读取任何图像")

    # 排序并构造 topk
    results.sort(key=lambda x: x[1], reverse=True)
    topk = [MatchCandidate(filename=f, good_matches=s) for (f, s, *_rest) in results[:topk_n]]

    best_name, best_score, kp_best, _des_best, img_best, pairs_best = results[0]
    best_id = _extract_id_from_name(best_name)

    # 生成可视化
    vis_path = None
    if save_vis:
        matched = cv2.drawMatchesKnn(
            src, kp_src, img_best, kp_best, pairs_best, None,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        )
        _ensure_dir(vis_output_path)
        cv2.imwrite(vis_output_path, matched)
        vis_path = vis_output_path

    ok = best_score >= not_found_threshold
    msg = (f"识别成功：ID={best_id} 文件={best_name} (good={best_score})"
           if ok else
           f"未识别：最高匹配数 {best_score} < 阈值 {not_found_threshold}")

    return IdentifyResult(ok, best_id if ok else None, best_name, best_score, topk, vis_path, msg)


