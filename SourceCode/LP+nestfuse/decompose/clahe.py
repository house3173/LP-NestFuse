import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim


# Tính PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Trường hợp hoàn hảo (không có sự khác biệt)
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


# Tính SSIM (Structural Similarity Index)
def calculate_ssim(img1, img2):
    return ssim(img1, img2)


# Tính Histogram Divergence
def histogram_divergence(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    # Thay thế các giá trị 0 trong histogram bằng 1e-6 để tránh chia cho 0
    hist1 = np.maximum(hist1, 1e-6)
    hist2 = np.maximum(hist2, 1e-6)

    divergence = np.sum(hist1 * np.log(hist1 / hist2))
    return divergence


# Tính Entropy của ảnh
def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()  # Normalize histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-6))  # Tránh log(0)
    return entropy


# Tính toán các tham số CLAHE động dựa trên đặc điểm ảnh
def calculate_clahe_params(img):
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)

    # Tính toán histogram của ảnh
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist / hist.sum()  # Chuẩn hóa histogram

    # Tính độ tương phản từ histogram
    contrast = np.sum((np.arange(256) - mean_intensity) ** 2 * hist_norm)

    # Quyết định clip_limit dựa trên độ tương phản và độ sáng trung bình
    if contrast < 5000:
        clip_limit = 3.0  # Giảm clip limit nếu ảnh có độ tương phản thấp
    elif contrast < 20000:
        clip_limit = 2.5  # Cải thiện một chút cho ảnh có độ tương phản trung bình
    else:
        clip_limit = 1.0  # Tăng clip limit cho ảnh có độ tương phản cao

    # Điều chỉnh grid size dựa trên kích thước ảnh
    height, width = img.shape
    grid_size = (max(8, width // 100), max(8, height // 100))

    return clip_limit, grid_size


# Lọc CLAHE và đánh giá chất lượng ảnh
def adaptive_clahe(img, iterations=5):
    best_psnr = -np.inf
    best_ssim = -np.inf
    best_divergence = np.inf
    best_entropy = -np.inf
    best_img = img.copy()

    for _ in range(iterations):
        # Tính toán tham số CLAHE linh động
        clip_limit, tile_grid_size = calculate_clahe_params(img)

        # Tạo đối tượng CLAHE và áp dụng
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_img = clahe.apply(img)

        # Tính các chỉ số đánh giá
        psnr_value = calculate_psnr(img, clahe_img)
        ssim_value = calculate_ssim(img, clahe_img)
        divergence_value = histogram_divergence(img, clahe_img)
        entropy_original = calculate_entropy(img)
        entropy_clahe = calculate_entropy(clahe_img)

        # Chỉ in log khi ảnh có chất lượng tốt hơn ảnh trước đó
        if psnr_value > best_psnr and ssim_value > best_ssim and divergence_value < best_divergence and entropy_clahe > best_entropy:
            best_psnr = psnr_value
            best_ssim = ssim_value
            best_divergence = divergence_value
            best_entropy = entropy_clahe
            best_img = clahe_img

            # # In thông tin log về chất lượng khi có sự cải thiện
            # print(f"Improvement found in iteration {_ + 1}:")
            # print(
            #     f"PSNR: {psnr_value}, SSIM: {ssim_value}, Divergence: {divergence_value}, Entropy (Original): {entropy_original}, Entropy (CLAHE): {entropy_clahe}")

    return best_img


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
            
            # Sử dụng tên gốc của ảnh mà không thay đổi
            output_path = os.path.join(output_folder, filename)

            # Lọc CLAHE và lưu ảnh đã xử lý
            adaptive_clahe(img_path, output_path)

            print(f"Processed image: {filename}, saved to: {output_path}")


# # Đường dẫn thư mục đầu vào và đầu ra
# input_folder = './code01/data01/MRI01'  # Thư mục chứa ảnh gốc
# output_folder = './code01/data01/MRI01Clahe'  # Thư mục lưu ảnh đã xử lý

# # Xử lý tất cả các ảnh trong thư mục đầu vào
# process_images_in_folder(input_folder, output_folder)
