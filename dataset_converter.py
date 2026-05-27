# dataset_converter.py - ForgeX 一鍵多選訓練集格式轉換工具（v2 加進度條 + 按鈕優化）
# 功能：多選 JSONL 文件 → 一鍵轉換到 Alpaca / ShareGPT / OpenAI Chat 格式
# 支持批量處理，自動生成新文件（原文件不變），實時進度條

import json
import os
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, ttk, StringVar, Label, Button, Frame
from typing import List, Dict
import threading

# ============================ 支持格式 ============================
FORMATS = {
    "Alpaca": {
        "desc": "標準指令格式 (instruction/input/output)",
        "convert": lambda item: {
            "instruction": item.get("instruction", item.get("prompt", "")),
            "input": item.get("input", ""),
            "output": item.get("output", item.get("response", ""))
        }
    },
    "ShareGPT": {
        "desc": "多輪對話格式 (messages 陣列)",
        "convert": lambda item: {
            "messages": [
                {"role": "user", "content": item.get("instruction", item.get("prompt", "")) + item.get("input", "")},
                {"role": "assistant", "content": item.get("output", item.get("response", ""))}
            ]
        }
    },
    "OpenAI Chat": {
        "desc": "OpenAI API 格式 (純 messages 陣列，無外層)",
        "convert": lambda item: [
            {"role": "user", "content": item.get("instruction", item.get("prompt", "")) + item.get("input", "")},
            {"role": "assistant", "content": item.get("output", item.get("response", ""))}
        ]
    }
}

# ============================ 轉換核心 ============================
def convert_file(input_path: Path, output_path: Path, target_format: str):
    """單文件轉換，返回樣本數"""
    converter = FORMATS[target_format]["convert"]
    samples = 0
    
    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                converted = converter(item)
                json.dump(converted, f_out, ensure_ascii=False, default=str)
                f_out.write("\n")
                samples += 1
            except json.JSONDecodeError:
                continue  # 跳過壞行
    
    return samples

# ============================ GUI 主體 ============================
def main():
    Tk().withdraw()  # 隱藏主窗口
    
    # 1. 多選文件
    messagebox.showinfo("選擇文件", "請按 Ctrl 多選需要轉換的 JSONL 文件（可跨文件夾）")
    input_files = filedialog.askopenfilenames(
        title="選擇訓練集 JSONL 文件",
        filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
    )
    
    if not input_files:
        messagebox.showinfo("完成", "未選擇文件，程序退出")
        return
    
    # 2. 主窗口
    root = Tk()
    root.title("ForgeX 訓練集格式轉換器 v2")
    root.geometry("500x400")
    root.resizable(False, False)
    
    Label(root, text="ForgeX 訓練集轉換工具", font=("Helvetica", 18, "bold")).pack(pady=20)
    Label(root, text=f"已選 {len(input_files)} 個文件", font=("Helvetica", 12)).pack(pady=5)
    
    # 格式選擇
    format_var = StringVar(value="Alpaca")
    Label(root, text="選擇目標格式：", font=("Helvetica", 12)).pack(pady=(15, 5))
    
    format_frame = Frame(root)
    format_frame.pack(pady=5)
    for fmt_name, fmt_info in FORMATS.items():
        ttk.Radiobutton(
            format_frame,
            text=f"{fmt_name} - {fmt_info['desc']}",
            variable=format_var,
            value=fmt_name
        ).pack(anchor="w")
    
    # 進度條
    progress_frame = Frame(root)
    progress_frame.pack(pady=20, fill="x", padx=40)
    progress_bar = ttk.Progressbar(progress_frame, mode="determinate")
    progress_bar.pack(fill="x")
    progress_label = Label(progress_frame, text="準備就緒", fg="blue")
    progress_label.pack(pady=5)
    
    # 轉換按鈕（大而明顯）
    convert_btn = Button(
        root, text="🔄 開始轉換（選擇輸出文件夾）", 
        font=("Helvetica", 14, "bold"), 
        bg="#4CAF50", fg="white", 
        height=2, width=30,
        command=lambda: threading.Thread(target=start_convert, daemon=True).start()
    )
    convert_btn.pack(pady=20)
    
    def start_convert():
        target_format = format_var.get()
        output_dir = filedialog.askdirectory(title="選擇輸出文件夾")
        if not output_dir:
            messagebox.showwarning("警告", "未選擇輸出文件夾")
            return
        
        output_dir = Path(output_dir)
        total_files = len(input_files)
        total_samples = 0
        success_count = 0
        
        progress_bar["maximum"] = total_files
        progress_bar["value"] = 0
        
        for i, input_path_str in enumerate(input_files):
            input_path = Path(input_path_str)
            output_name = f"{input_path.stem}_{target_format.lower()}{input_path.suffix}"
            output_path = output_dir / output_name
            
            progress_label.config(text=f"處理中: {input_path.name} ({i+1}/{total_files})")
            root.update()
            
            try:
                samples = convert_file(input_path, output_path, target_format)
                total_samples += samples
                success_count += 1
            except Exception as e:
                progress_label.config(text=f"錯誤: {input_path.name} - {e}", fg="red")
                root.update()
                continue
            
            progress_bar["value"] = i + 1
            root.update()
        
        progress_label.config(text="轉換完成！", fg="green")
        message = f"✅ 轉換完成！\n成功 {success_count}/{total_files} 個文件\n總計 {total_samples} 條樣本\n輸出目錄: {output_dir}"
        messagebox.showinfo("成功", message)
        convert_btn.config(state="normal")
    
    root.mainloop()

if __name__ == "__main__":
    main()
