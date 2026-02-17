"""
Tkinter Multi-Character Handwriting Recognition Test Client
Provides a drawing area for testing the multi-character recognition API
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
import io
import base64
import requests
import json
import os
import numpy as np


class MultiCharRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多字手写识别测试")
        self.root.geometry("1200x800")
        
        # API configuration
        self.api_base_url = "https://api.aozai.top/InkForge"
        
        # Drawing variables
        self.canvas_width = 800
        self.canvas_height = 200
        self.brush_size = 15
        
        # Use black color for strokes (testing preprocessing)
        # Tkinter uses hex string for display on canvas
        self.tk_color = '#000000'  # Black color for display (hex) - RGB(0, 0, 0)
        # PIL uses int for grayscale image (0=black, 255=white)
        # Convert RGB(0, 0, 0) to grayscale: 0.299*0 + 0.587*0 + 0.114*0 = 0
        self.pil_color = 0  # Black value for internal grayscale image
        
        self.last_x = None
        self.last_y = None
        
        # Image for drawing - use L (grayscale) mode to match training data
        # This ensures black strokes (0) on white background (255) for testing preprocessing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # Result image with bounding boxes
        self.result_image = None
        self.result_photo = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="多字手写识别测试", font=("Arial", 16, "bold"))
        title_label.pack(pady=5)
        
        # Drawing area frame
        draw_frame = ttk.LabelFrame(main_frame, text="手写区域 (请在下方书写多个汉字)", padding="5")
        draw_frame.pack(fill=tk.X, pady=5)
        
        # Canvas for drawing
        self.canvas = tk.Canvas(
            draw_frame, 
            width=self.canvas_width, 
            height=self.canvas_height,
            bg="white",
            cursor="cross"
        )
        self.canvas.pack(pady=5)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Button frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        # Buttons
        ttk.Button(btn_frame, text="清除", command=self.clear_canvas, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="识别 (多字)", command=self.recognize_multi, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="识别 (单字)", command=self.recognize_single, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存图片", command=self.save_image, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="文件识别(单字)", command=self.recognize_from_file, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="文件识别(多字)", command=self.recognize_multi_from_file, width=15).pack(side=tk.LEFT, padx=5)
        
        # Brush size
        ttk.Label(btn_frame, text="笔刷大小:").pack(side=tk.LEFT, padx=(20, 5))
        self.brush_size_var = tk.IntVar(value=15)
        brush_scale = ttk.Scale(btn_frame, from_=5, to=30, variable=self.brush_size_var, 
                                orient=tk.HORIZONTAL, length=100)
        brush_scale.pack(side=tk.LEFT, padx=5)
        
        # API URL
        ttk.Label(btn_frame, text="API地址:").pack(side=tk.LEFT, padx=(20, 5))
        self.api_url_var = tk.StringVar(value=self.api_base_url)
        api_entry = ttk.Entry(btn_frame, textvariable=self.api_url_var, width=30)
        api_entry.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="识别结果", padding="5")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Result text
        self.result_text = tk.Text(result_frame, height=8, wrap=tk.WORD)
        self.result_text.pack(fill=tk.X, pady=5)
        
        # Result image frame
        img_frame = ttk.LabelFrame(main_frame, text="识别框位置可视化", padding="5")
        img_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Result image label
        self.result_img_label = ttk.Label(img_frame)
        self.result_img_label.pack(pady=5)
        
        # Preprocessed images frame
        preproc_frame = ttk.LabelFrame(main_frame, text="预处理图像预览 (模型实际看到的图像)", padding="5")
        preproc_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas for preprocessed images (scrollable)
        self.preproc_canvas = tk.Canvas(preproc_frame, height=100)
        self.preproc_scrollbar = ttk.Scrollbar(preproc_frame, orient=tk.HORIZONTAL, command=self.preproc_canvas.xview)
        self.preproc_inner_frame = ttk.Frame(self.preproc_canvas)
        
        self.preproc_canvas.configure(xscrollcommand=self.preproc_scrollbar.set)
        self.preproc_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.preproc_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.preproc_canvas_window = self.preproc_canvas.create_window((0, 0), window=self.preproc_inner_frame, anchor=tk.NW)
        self.preproc_inner_frame.bind("<Configure>", self.on_frame_configure)
        
        # Store preprocessed image references
        self.preproc_photos = []
        
        # Status bar
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=5)
        
    def start_draw(self, event):
        """Start drawing"""
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        """Draw a line with gray color (matching training data)"""
        if self.last_x is not None and self.last_y is not None:
            w = self.brush_size_var.get()
            
            # Draw on canvas (Tkinter display)
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=w,
                fill=self.tk_color,
                capstyle=tk.ROUND,
                smooth=True
            )
            
            # Draw on PIL image (internal)
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=self.pil_color,
                width=w
            )
            
            # Draw circles at endpoints for smooth rounded strokes
            r = w / 2
            self.draw.ellipse(
                (self.last_x - r, self.last_y - r, self.last_x + r, self.last_y + r),
                fill=self.pil_color
            )
            self.draw.ellipse(
                (event.x - r, event.y - r, event.x + r, event.y + r),
                fill=self.pil_color
            )
            
            self.last_x = event.x
            self.last_y = event.y
            
    def stop_draw(self, event):
        """Stop drawing"""
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_text.delete(1.0, tk.END)
        self.result_img_label.config(image="")
        self.result_photo = None
        
        # Clear preprocessed images
        for widget in self.preproc_inner_frame.winfo_children():
            widget.destroy()
        self.preproc_photos = []
        
        self.status_var.set("已清除")
        
    def on_frame_configure(self, event):
        """Update scroll region when inner frame changes"""
        self.preproc_canvas.configure(scrollregion=self.preproc_canvas.bbox("all"))
        
    def image_to_base64(self):
        """Convert the current image to base64"""
        # Convert to grayscale for API
        gray_image = self.image.convert("L")
        
        # Convert to bytes
        buffer = io.BytesIO()
        gray_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return base64_str
    
    def recognize_multi(self):
        """Send image to multi-character recognition API"""
        self.status_var.set("正在识别...")
        self.root.update()
        
        try:
            # Get base64 image
            base64_image = self.image_to_base64()
            
            # Prepare request
            api_url = self.api_url_var.get()
            url = f"{api_url}/multipredict/base64"
            
            payload = {"image": base64_image}
            
            # Send request
            response = requests.post(
                url,
                json=payload,
                params={"model_name": "multi_char_handwriting", "top_k": 5},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.display_result(result)
                self.draw_bounding_boxes(result)
                self.status_var.set("识别完成")
            else:
                error_msg = response.json().get("detail", "未知错误")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"错误: {error_msg}\n")
                self.status_var.set(f"错误: {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "错误: 请求超时\n")
            self.status_var.set("请求超时")
        except requests.exceptions.ConnectionError:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "错误: 无法连接到服务器\n")
            self.status_var.set("连接失败")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"错误: {str(e)}\n")
            self.status_var.set("发生错误")
            
    def save_image(self):
        """Save current image to file"""
        try:
            # Create filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"handwriting_{timestamp}.png"

            # Save the image
            self.image.save(filename)

            self.status_var.set(f"图片已保存: {filename}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"图片已保存到: {filename}\n")
            self.result_text.insert(tk.END, f"文件大小: {os.path.getsize(filename)} bytes\n")

            # Show image info
            img_array = np.array(self.image.convert("L"))
            self.result_text.insert(tk.END, f"图像尺寸: {img_array.shape}\n")
            self.result_text.insert(tk.END, f"像素值范围: [{np.min(img_array)}, {np.max(img_array)}]\n")
            self.result_text.insert(tk.END, f"平均值: {np.mean(img_array):.2f}\n")

            unique_values = np.unique(img_array)
            if len(unique_values) <= 10:
                self.result_text.insert(tk.END, f"唯一值: {unique_values}\n")
            else:
                self.result_text.insert(tk.END, f"唯一值数量: {len(unique_values)}\n")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"保存失败: {str(e)}\n")
            self.status_var.set("保存失败")

    def recognize_from_file(self):
        """Recognize from saved image file using file upload (single character)"""
        self.status_var.set("选择文件中...")
        self.root.update()

        try:
            # Ask for file
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(
                title="选择手写图片",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
            )

            if not file_path:
                self.status_var.set("已取消")
                return

            self.status_var.set("正在识别文件...")
            self.root.update()

            # Prepare request
            api_url = self.api_url_var.get()
            url = f"{api_url}/predict"

            # Read file
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'image/png')}

                # Send request
                response = requests.post(
                    url,
                    files=files,
                    params={"model_name": "chinese_handwriting", "top_k": 5},
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                self.display_single_result(result)
                self.status_var.set(f"文件识别完成: {os.path.basename(file_path)}")
            else:
                error_msg = response.json().get("detail", "未知错误")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"错误: {error_msg}\n")
                self.status_var.set(f"错误: {response.status_code}")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"错误: {str(e)}\n")
            self.status_var.set("发生错误")

    def recognize_multi_from_file(self):
        """Recognize from saved image file using file upload (multi-character)"""
        self.status_var.set("选择文件中...")
        self.root.update()

        try:
            # Ask for file
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(
                title="选择手写图片",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
            )

            if not file_path:
                self.status_var.set("已取消")
                return

            self.status_var.set("正在识别文件...")
            self.root.update()

            # Prepare request
            api_url = self.api_url_var.get()
            url = f"{api_url}/multipredict"

            # Read file
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'image/png')}

                # Send request
                response = requests.post(
                    url,
                    files=files,
                    params={"model_name": "multi_char_handwriting", "top_k": 5},
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                self.display_result(result)
                self.draw_bounding_boxes(result)
                self.status_var.set(f"文件识别完成: {os.path.basename(file_path)}")
            else:
                error_msg = response.json().get("detail", "未知错误")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"错误: {error_msg}\n")
                self.status_var.set(f"错误: {response.status_code}")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"错误: {str(e)}\n")
            self.status_var.set("发生错误")

    def recognize_single(self):
        """Send image to single character recognition API"""
        self.status_var.set("正在识别...")
        self.root.update()

        try:
            # Get base64 image
            base64_image = self.image_to_base64()

            # Prepare request
            api_url = self.api_url_var.get()
            url = f"{api_url}/predict/base64"

            payload = {"image": base64_image}

            # Send request
            response = requests.post(
                url,
                json=payload,
                params={"model_name": "chinese_handwriting", "top_k": 5},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                self.display_single_result(result)
                self.status_var.set("识别完成")
            else:
                error_msg = response.json().get("detail", "未知错误")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"错误: {error_msg}\n")
                self.status_var.set(f"错误: {response.status_code}")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"错误: {str(e)}\n")
            self.status_var.set("发生错误")
            
    def display_result(self, result):
        """Display multi-character recognition result"""
        self.result_text.delete(1.0, tk.END)
        
        text = result.get("text", "")
        success = result.get("success", False)
        num_chars = result.get("num_characters", 0)
        num_blocks = result.get("num_blocks", 0)
        total_cost = result.get("total_cost", 0)
        timing = result.get("timing", {})
        characters = result.get("characters", [])
        
        # Display summary
        self.result_text.insert(tk.END, f"识别结果: {text}\n")
        self.result_text.insert(tk.END, f"成功: {success}\n")
        self.result_text.insert(tk.END, f"字符数: {num_chars}, 原子块数: {num_blocks}\n")
        self.result_text.insert(tk.END, f"总代价: {total_cost:.4f}\n")
        
        # Display timing
        total_time = sum(timing.values()) if timing else 0
        self.result_text.insert(tk.END, f"总耗时: {total_time*1000:.2f}ms\n")
        self.result_text.insert(tk.END, "-" * 40 + "\n")
        
        # Display character details
        self.result_text.insert(tk.END, "字符详情:\n")
        for i, char in enumerate(characters):
            char_str = char.get("character", "?")
            conf = char.get("confidence", 0)
            x_start = char.get("x_start", 0)
            x_end = char.get("x_end", 0)
            y_start = char.get("y_start", 0)
            y_end = char.get("y_end", 0)
            num_blocks = char.get("num_blocks", 1)
            
            self.result_text.insert(tk.END, 
                f"  [{i+1}] '{char_str}' 置信度: {conf:.2f} "
                f"位置: ({x_start},{y_start})-({x_end},{y_end}) "
                f"块数: {num_blocks}\n"
            )
            
            # Display top predictions
            predictions = char.get("predictions", [])
            if predictions:
                self.result_text.insert(tk.END, f"      候选: ")
                for pred in predictions[:3]:
                    self.result_text.insert(tk.END, f"{pred.get('character', '?')}({pred.get('confidence', 0):.2f}) ")
                self.result_text.insert(tk.END, "\n")
        
        # Display preprocessed images
        self.display_preprocessed_images(characters)
                
    def display_single_result(self, result):
        """Display single character recognition result"""
        self.result_text.delete(1.0, tk.END)
        
        predictions = result.get("predictions", [])
        
        if predictions:
            best = predictions[0]
            self.result_text.insert(tk.END, f"识别结果: {best.get('character', '?')}\n")
            self.result_text.insert(tk.END, f"置信度: {best.get('confidence', 0):.4f}\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
            self.result_text.insert(tk.END, "候选字符:\n")
            
            for i, pred in enumerate(predictions):
                char = pred.get("character", "?")
                conf = pred.get("confidence", 0)
                self.result_text.insert(tk.END, f"  [{i+1}] '{char}' 置信度: {conf:.4f}\n")
        else:
            self.result_text.insert(tk.END, "未识别到字符\n")
            
        # Clear bounding box visualization
        self.result_img_label.config(image="")
        self.result_photo = None
        
        # Clear preprocessed images
        for widget in self.preproc_inner_frame.winfo_children():
            widget.destroy()
        self.preproc_photos = []
        
    def display_preprocessed_images(self, characters):
        """Display preprocessed images for each character"""
        # Clear previous images
        for widget in self.preproc_inner_frame.winfo_children():
            widget.destroy()
        self.preproc_photos = []
        
        for i, char in enumerate(characters):
            img_base64 = char.get("image_base64")
            if not img_base64:
                continue
                
            try:
                # Decode base64 image
                img_bytes = base64.b64decode(img_base64)
                img = Image.open(io.BytesIO(img_bytes))
                
                # Resize for display (64x64 is too small, scale up)
                img = img.resize((64, 64), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                self.preproc_photos.append(photo)  # Keep reference
                
                # Create frame for this image
                frame = ttk.Frame(self.preproc_inner_frame)
                frame.pack(side=tk.LEFT, padx=5, pady=5)
                
                # Image label
                img_label = ttk.Label(frame, image=photo)
                img_label.pack()
                
                # Character label
                char_str = char.get("character", "?")
                conf = char.get("confidence", 0)
                char_label = ttk.Label(frame, text=f"'{char_str}' ({conf:.2f})")
                char_label.pack()
                
            except Exception as e:
                print(f"Error displaying preprocessed image: {e}")
            
    def draw_bounding_boxes(self, result):
        """Draw bounding boxes on the original image"""
        characters = result.get("characters", [])
        
        if not characters:
            return
            
        # Make a copy of the original image
        result_img = self.image.copy()
        draw = ImageDraw.Draw(result_img)
        
        # Colors for different characters
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
            (0, 128, 128),    # Teal
        ]
        
        for i, char in enumerate(characters):
            x_start = char.get("x_start", 0)
            x_end = char.get("x_end", 0)
            y_start = char.get("y_start", 0)
            y_end = char.get("y_end", 0)
            character = char.get("character", "?")
            confidence = char.get("confidence", 0)
            
            color = colors[i % len(colors)]
            
            # Draw rectangle
            draw.rectangle(
                [x_start, y_start, x_end, y_end],
                outline=color,
                width=3
            )
            
            # Draw character label above the box
            label = f"{character}({confidence:.2f})"
            # Position text above the box
            text_y = max(0, y_start - 20)
            draw.text((x_start, text_y), label, fill=color)
            
        # Convert to PhotoImage
        # Resize if needed to fit display
        max_width = 800
        max_height = 200
        img_width, img_height = result_img.size
        
        if img_width > max_width or img_height > max_height:
            scale = min(max_width / img_width, max_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            result_img = result_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.result_photo = ImageTk.PhotoImage(result_img)
        self.result_img_label.config(image=self.result_photo)


def main():
    root = tk.Tk()
    app = MultiCharRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
