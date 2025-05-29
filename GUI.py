import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
from train import IMG_SIZE

class LungXrayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chẩn Đoán Ung Thư Phổi qua X-ray")
        self.root.state('zoomed') 
    
        # Lấy kích thước màn hình
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        
        # Thiết lập màu nền
        self.root.configure(bg='#f0f0f0')
        
        # Tạo style
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 24, "bold"), background="#f0f0f0", foreground="#2c3e50")
        style.configure("Result.TLabel", font=("Helvetica", 14), background="#ffffff", foreground="#2c3e50")
        style.configure("Warning.TLabel", font=("Helvetica", 14, "bold"), background="#ffffff", foreground="#e74c3c")
        style.configure("Normal.TLabel", font=("Helvetica", 14, "bold"), background="#ffffff", foreground="#27ae60")
        style.configure("Custom.TButton", font=("Helvetica", 12, "bold"), padding=15)
        
        # Main Frame với padding tự động điều chỉnh
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=True, fill="both", padx=int(self.screen_width*0.1), pady=int(self.screen_height*0.05))
        
        # Title với icon
        title = ttk.Label(
            self.main_frame,
            text="CHẨN ĐOÁN UNG THƯ PHỔI QUA ẢNH X-RAY",
            style="Title.TLabel"
        )
        title.pack(pady=int(self.screen_height*0.03))
        
        # Frame cho ảnh với viền
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(pady=int(self.screen_height*0.02))
        
        # Label để hiển thị ảnh
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()
        
        # Frame cho các nút
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=int(self.screen_height*0.02))
        
        # Nút chọn ảnh
        self.select_button = ttk.Button(
            button_frame,
            text="Chọn ảnh X-ray",
            command=self.select_image,
            style="Custom.TButton"
        )
        self.select_button.pack(side=tk.LEFT, padx=20)
        
        # Nút dự đoán
        self.predict_button = ttk.Button(
            button_frame,
            text="Chẩn đoán",
            command=self.predict,
            state=tk.DISABLED,
            style="Custom.TButton"
        )
        self.predict_button.pack(side=tk.LEFT, padx=20)
        
        # Frame cho kết quả với nền trắng và viền
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.pack(pady=int(self.screen_height*0.02), fill="x", padx=int(self.screen_width*0.2))
        
        # Label cho xác suất
        self.probability_label = ttk.Label(
            self.result_frame,
            text="",
            style="Result.TLabel",
            wraplength=int(self.screen_width*0.6)
        )
        self.probability_label.pack(pady=(20, 10))
        
        # Label cho chẩn đoán
        self.diagnosis_label = ttk.Label(
            self.result_frame,
            text="",
            style="Normal.TLabel",
            wraplength=int(self.screen_width*0.6)
        )
        self.diagnosis_label.pack(pady=10)
        
        # Label cho khuyến nghị
        self.recommendation_label = ttk.Label(
            self.result_frame,
            text="",
            style="Result.TLabel",
            wraplength=int(self.screen_width*0.6)
        )
        self.recommendation_label.pack(pady=(10, 20))
        
        # Khởi tạo các biến
        self.model = None
        self.current_image = None
        self.image_path = None
        
        # Load model
        try:
            self.model = tf.keras.models.load_model('lung_xray_model.h5')
            print("Đã tải mô hình thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh X-ray",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                image = Image.open(file_path)
                
                # Tính toán kích thước hiển thị dựa trên màn hình
                display_size = (int(self.screen_width*0.4), int(self.screen_height*0.4))
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Tạo khung trắng cho ảnh
                background = Image.new('RGB', display_size, 'white')
                # Tính toán vị trí để căn giữa ảnh
                x = (display_size[0] - image.size[0]) // 2
                y = (display_size[1] - image.size[1]) // 2
                background.paste(image, (x, y))
                
                photo = ImageTk.PhotoImage(background)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                
                self.predict_button.configure(state=tk.NORMAL)
                self.clear_results()
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể mở ảnh: {str(e)}")
    
    def clear_results(self):
        self.probability_label.configure(text="")
        self.diagnosis_label.configure(text="")
        self.recommendation_label.configure(text="")
    
    def predict(self):
        if self.image_path and self.model:
            try:
                img = tf.keras.preprocessing.image.load_img(
                    self.image_path,
                    target_size=(IMG_SIZE, IMG_SIZE)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0
                
                prediction = self.model.predict(img_array, verbose=0)[0][0]
                
                # Hiển thị xác suất
                self.probability_label.configure(
                    text=f"Xác suất ung thư phổi: {prediction*100:.2f}%",
                    style="Result.TLabel"
                )
                
                # Hiển thị chẩn đoán và khuyến nghị
                if prediction >= 0.5:
                    self.diagnosis_label.configure(
                        text="CẢNH BÁO: Có dấu hiệu bất thường!",
                        style="Warning.TLabel"
                    )
                    self.recommendation_label.configure(
                        text="Vui lòng đến gặp bác sĩ để kiểm tra chi tiết."
                    )
                else:
                    self.diagnosis_label.configure(
                        text="Không phát hiện dấu hiệu bất thường.",
                        style="Normal.TLabel"
                    )
                    self.recommendation_label.configure(
                        text="Tiếp tục theo dõi sức khỏe định kỳ."
                    )
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi dự đoán: {str(e)}")
        else:
            messagebox.showerror("Lỗi", "Vui lòng chọn ảnh và đảm bảo mô hình đã được tải.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LungXrayApp(root)
    root.mainloop()
