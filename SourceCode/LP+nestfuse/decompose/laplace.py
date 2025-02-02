import cv2
import numpy as np
import os

# Tính độ tương phản của ảnh (trung bình độ chênh lệch giữa các pixel)
def calculate_contrast(img):
    return np.std(img)

# Tính entropy của ảnh (mức độ thông tin hoặc chi tiết của ảnh)
def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()  # Normalize histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-6))  # Tránh log(0)
    return entropy

# Hàm làm sắc nét với bộ lọc Laplace và tham số động
def laplacian_sharpening_dynamic(img):
    # Tính độ tương phản và entropy
    contrast = calculate_contrast(img)
    entropy = calculate_entropy(img)

    # Điều chỉnh tham số sắc nét dựa trên độ tương phản và entropy
    if contrast < 30:
        weight = 1.2  # Tăng mức độ sắc nét cho ảnh ít tương phản
        laplace_weight = -0.4
    elif contrast < 60:
        weight = 1.4  # Độ sắc nét vừa phải cho ảnh có độ tương phản trung bình
        laplace_weight = -0.5
    else:
        weight = 1.6  # Tăng mạnh mức độ sắc nét cho ảnh có độ tương phản cao
        laplace_weight = -0.6

    # Nếu entropy thấp, có thể là ảnh quá tối hoặc quá sáng, ta giảm sắc nét
    if entropy < 5.0:
        weight = 1.2
        laplace_weight = -0.3

    # Áp dụng bộ lọc Laplace
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # Chuyển đổi giá trị về dạng uint8 (giá trị pixel)
    laplacian = cv2.convertScaleAbs(laplacian)
    
    # Thêm kết quả Laplace vào ảnh gốc để làm sắc nét
    sharpened_img = cv2.addWeighted(img, weight, laplacian, laplace_weight, 0)
    
    return sharpened_img


# Lọc tất cả các ảnh trong thư mục đầu vào và lưu vào thư mục đầu ra
def process_images_in_folder(input_folder, output_folder):
    # Kiểm tra nếu thư mục đầu ra không tồn tại, tạo mới
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lấy tất cả các tệp trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        # Kiểm tra nếu tệp là ảnh (theo định dạng .png, .jpg, .jpeg, v.v.)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            
            # Đọc ảnh gốc
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Áp dụng bộ lọc Laplace và làm sắc nét ảnh
            sharpened_img = laplacian_sharpening_dynamic(img)
            
            # Tạo đường dẫn lưu ảnh đã qua xử lý
            output_path = os.path.join(output_folder, filename)
            
            # Lưu ảnh đã xử lý
            cv2.imwrite(output_path, sharpened_img)

            print(f"Processed image: {filename}, saved to: {output_path}")


# # Đường dẫn thư mục đầu vào và đầu ra
# input_folder = './code01/data01/MRI01Clahe'  # Thư mục chứa ảnh gốc
# output_folder = './code01/data01/MRI01ClaheLaplace'  # Thư mục lưu ảnh đã xử lý

# # Xử lý tất cả các ảnh trong thư mục đầu vào
# process_images_in_folder(input_folder, output_folder)
