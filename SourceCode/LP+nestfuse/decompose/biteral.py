import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

# Hàm để tính toán các tham số tự động cho bộ lọc Bilateral
def calculate_bilateral_parameters(image):
    """
    Tính toán các tham số d, sigmaColor, sigmaSpace cho bộ lọc Bilateral dựa trên kích thước và độ tương phản của ảnh.
    """
    # Tính toán các thông số dựa trên kích thước ảnh
    height, width = image.shape

    # d: Đường kính của vùng lân cận (theo tỷ lệ với kích thước ảnh)
    d = int(min(height, width) / 20)  # Điều chỉnh tỷ lệ này tùy thuộc vào ảnh của bạn

    # sigmaColor và sigmaSpace có thể được tính toán dựa trên độ tương phản của ảnh
    contrast = np.std(image)  # Độ lệch chuẩn (standard deviation) của ảnh

    # Chọn giá trị sigmaColor và sigmaSpace dựa trên độ tương phản
    sigmaColor = contrast * 0.5  # Điều chỉnh giá trị này tùy theo độ tương phản
    sigmaSpace = contrast * 0.4  # Điều chỉnh giá trị này tùy theo độ tương phản

    # Đảm bảo các tham số nằm trong một phạm vi hợp lý
    sigmaColor = max(10, min(sigmaColor, 150))
    sigmaSpace = max(10, min(sigmaSpace, 150))

    return d, sigmaColor, sigmaSpace


# Hàm để xử lý tất cả ảnh trong một thư mục và lưu kết quả vào thư mục mới
def process_mri_images(input_folder, output_folder_detail, output_folder_base):
    # Kiểm tra và tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder_detail, exist_ok=True)
    os.makedirs(output_folder_base, exist_ok=True)

    # Lấy tất cả các tệp ảnh trong thư mục
    image_paths = glob(os.path.join(input_folder, '*.png')) + glob(os.path.join(input_folder, '*.jpg'))

    # Duyệt qua tất cả các ảnh trong thư mục
    for image_path in image_paths:
        # Đọc ảnh MRI
        mri_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Kiểm tra xem ảnh có đọc thành công không
        if mri_image is None:
            print(f"Không thể đọc ảnh {image_path}. Vui lòng kiểm tra lại đường dẫn ảnh!")
            continue

        # Bước 2: Tính toán các tham số tự động cho bộ lọc Bilateral
        d, sigmaColor, sigmaSpace = calculate_bilateral_parameters(mri_image)

        # Áp dụng bộ lọc Bilateral lên ảnh MRI
        filtered_image = cv2.bilateralFilter(mri_image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

        # Bước 4: Tính thành phần chi tiết (Ics = I - Ict) và thành phần cơ sở (Ict)
        base_image = filtered_image  # Thành phần chi tiết là ảnh đã lọc (Ict)
        detail_image = cv2.subtract(mri_image, base_image)  # Thành phần cơ sở là ảnh gốc trừ thành phần chi tiết (Ics)

        # Lưu ảnh chi tiết và ảnh cơ sở vào các thư mục đầu ra với hậu tố "_detail" và "_base"
        detail_image_path = os.path.join(output_folder_detail, os.path.splitext(os.path.basename(image_path))[0] + '.png')
        base_image_path = os.path.join(output_folder_base, os.path.splitext(os.path.basename(image_path))[0] + '.png')

        # Lưu ảnh chi tiết và cơ sở
        cv2.imwrite(detail_image_path, detail_image)
        cv2.imwrite(base_image_path, base_image)

        print(f"Đã xử lý và lưu ảnh {os.path.basename(image_path)} vào thư mục {output_folder_detail} và {output_folder_base}.")

    

# # Đường dẫn tới thư mục chứa ảnh MRI gốc và thư mục lưu ảnh đã xử lý
# input_folder = './code01/data01/MRI01ClaheLaplace'  # Đường dẫn tới thư mục ảnh MRI gốc
# output_folder_detail = './code01/data01/MRI01detail'  # Thư mục lưu ảnh chi tiết
# output_folder_base = './code01/data01/MRI01base'  # Thư mục lưu ảnh cơ sở

# # Gọi hàm để xử lý ảnh trong thư mục
# process_mri_images(input_folder, output_folder_detail, output_folder_base)
