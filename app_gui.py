# app_gui.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from fingerprint_core import identify_in_db, IdentifyResult

APP_TITLE = "Fingerprint Identifier (Tk)"
DEFAULT_RATIO = 0.80
DEFAULT_THRESHOLD = 15

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x620")
        self.minsize(900, 560)

        # 选择路径区
        self.src_path_var = tk.StringVar(value="data/src.bmp")
        self.db_dir_var  = tk.StringVar(value="data/database")
        self.ratio_var   = tk.StringVar(value=str(DEFAULT_RATIO))
        self.thres_var   = tk.StringVar(value=str(DEFAULT_THRESHOLD))

        self._build_left_controls()
        self._build_right_panel()

        self.preview_img_tk = None   # 预览用引用（避免被垃圾回收）
        self.vis_img_tk = None

    def _build_left_controls(self):
        frm = tk.Frame(self, padx=12, pady=12)
        frm.pack(side=tk.LEFT, fill=tk.Y)

        # 输入图
        tk.Label(frm, text="输入指纹图 (src)").pack(anchor="w")
        row1 = tk.Frame(frm); row1.pack(fill=tk.X, pady=4)
        tk.Entry(row1, textvariable=self.src_path_var, width=42).pack(side=tk.LEFT, padx=(0,6))
        tk.Button(row1, text="选择图片", command=self.pick_src).pack(side=tk.LEFT)

        # 数据库目录
        tk.Label(frm, text="数据库目录 (database)").pack(anchor="w", pady=(10,0))
        row2 = tk.Frame(frm); row2.pack(fill=tk.X, pady=4)
        tk.Entry(row2, textvariable=self.db_dir_var, width=42).pack(side=tk.LEFT, padx=(0,6))
        tk.Button(row2, text="选择目录", command=self.pick_db).pack(side=tk.LEFT)

        # 参数
        tk.Label(frm, text="参数").pack(anchor="w", pady=(10,0))
        row3 = tk.Frame(frm); row3.pack(fill=tk.X, pady=4)
        tk.Label(row3, text="ratio").pack(side=tk.LEFT)
        tk.Entry(row3, textvariable=self.ratio_var, width=8).pack(side=tk.LEFT, padx=(4,12))
        tk.Label(row3, text="未识别阈值").pack(side=tk.LEFT)
        tk.Entry(row3, textvariable=self.thres_var, width=8).pack(side=tk.LEFT, padx=(4,12))

        # 操作
        tk.Button(frm, text="识别", width=16, command=self.run_identify).pack(pady=(16,6))
        tk.Button(frm, text="打开可视化文件夹", command=self.open_output_folder).pack()

        # 预览输入图
        tk.Label(frm, text="输入图预览").pack(anchor="w", pady=(16,6))
        self.canvas_src = tk.Canvas(frm, width=280, height=280, bg="#f0f0f0")
        self.canvas_src.pack()

        tk.Button(frm, text="刷新预览", command=self.refresh_src_preview).pack(pady=6)

    def _build_right_panel(self):
        frm = tk.Frame(self, padx=12, pady=12)
        frm.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(frm, text="识别结果").pack(anchor="w")
        self.txt_result = tk.Text(frm, height=6)
        self.txt_result.pack(fill=tk.X, pady=(4,12))

        tk.Label(frm, text="最佳匹配可视化").pack(anchor="w")
        self.canvas_vis = tk.Canvas(frm, bg="#f8f8f8")
        self.canvas_vis.pack(fill=tk.BOTH, expand=True)

    def pick_src(self):
        path = filedialog.askopenfilename(
            title="选择输入指纹图",
            filetypes=[("Image", "*.bmp *.png *.jpg *.jpeg *.BMP *.PNG *.JPG *.JPEG"), ("All files", "*.*")]
        )
        if path:
            self.src_path_var.set(path)
            self.refresh_src_preview()

    def pick_db(self):
        path = filedialog.askdirectory(title="选择数据库目录")
        if path:
            self.db_dir_var.set(path)

    def refresh_src_preview(self):
        path = self.src_path_var.get().strip()
        if not os.path.exists(path):
            messagebox.showwarning("提示", f"找不到图片：{path}")
            return
        try:
            img = Image.open(path).convert("L")
            img.thumbnail((280, 280))
            self.preview_img_tk = ImageTk.PhotoImage(img)
            self.canvas_src.delete("all")
            self.canvas_src.create_image(140, 140, image=self.preview_img_tk)
        except Exception as e:
            messagebox.showerror("错误", f"预览失败：{e}")

    def run_identify(self):
        # 参数解析
        src = self.src_path_var.get().strip()
        db  = self.db_dir_var.get().strip()
        try:
            ratio = float(self.ratio_var.get().strip())
            thres = int(float(self.thres_var.get().strip()))
        except:
            messagebox.showerror("错误", "参数格式不正确（ratio 用小数，阈值用整数）")
            return

        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, "正在识别，请稍候...\n")
        self.update_idletasks()

        # 调用核心接口
        result: IdentifyResult = identify_in_db(
            src_path=src,
            db_dir=db,
            ratio=ratio,
            not_found_threshold=thres,
            topk_n=3,
            save_vis=True,
            vis_output_path="output/best_match.png",
            preview_scale=3
        )

        # 展示文本结果
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, result.message + "\n\n")
        if result.topk:
            self.txt_result.insert(tk.END, "Top 候选：\n")
            for c in result.topk:
                self.txt_result.insert(tk.END, f"  {c.filename:25s}  good={c.good_matches}\n")

        # 展示可视化图
        self.show_vis_image(result.vis_path)

    def show_vis_image(self, vis_path: str):
        self.canvas_vis.delete("all")
        if not vis_path or not os.path.exists(vis_path):
            self.txt_result.insert(tk.END, "\n未生成可视化图片\n")
            return
        try:
            # 根据右侧画布大小自适应缩放
            cw = self.canvas_vis.winfo_width() or 800
            ch = self.canvas_vis.winfo_height() or 360
            img = Image.open(vis_path).convert("RGB")
            img.thumbnail((cw, ch))
            self.vis_img_tk = ImageTk.PhotoImage(img)
            self.canvas_vis.create_image(cw//2, ch//2, image=self.vis_img_tk)
        except Exception as e:
            self.txt_result.insert(tk.END, f"\n可视化加载失败：{e}\n")

    def open_output_folder(self):
        out_dir = os.path.abspath("output")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(out_dir)  # Windows
            elif os.uname().sysname == "Darwin":
                os.system(f"open '{out_dir}'")  # macOS
            else:
                os.system(f"xdg-open '{out_dir}'")  # Linux
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录：{e}")

if __name__ == "__main__":
    app = App()
    app.after(200, app.refresh_src_preview)  # 初次加载试着预览默认路径
    app.mainloop()
