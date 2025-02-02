import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
from scipy.ndimage import laplace
import pywt
import dtcwt
import cv2
from decompose.clahe import adaptive_clahe
from decompose.laplace import laplacian_sharpening_dynamic

# Fusion of two images (grayscale)

def gausspyr_reduce(x, kernel_a=0.4):
    # Kernel
    K = np.array([0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2])
    
    x = x.reshape(x.shape[0], x.shape[1], -1) # Add an extra dimension if grayscale
    y = np.zeros([math.ceil(x.shape[0]/2), math.ceil(x.shape[1]/2), x.shape[2]]) # Store the result in this array
    
    for cc in range(x.shape[2]): # for each colour channel
        # Step 1: filter rows
        y_a = sp.signal.convolve2d(x[:,:,cc], K.reshape(1,-1), mode='same', boundary='symm')
        # Step 2: subsample rows (skip every second column)
        # <-------------------- ???????? y_a = y_a[:,::2] ????????? ------------------> #
        y_a = x[:,::2,cc]   
        # Step 3: filter columns
        y_a = sp.signal.convolve2d(y_a, K.reshape(-1,1), mode='same', boundary='symm')
        # Step 4: subsample columns (skip every second row)
        y[:,:,cc] = y_a[::2,:]
    return np.squeeze(y) # remove an extra dimension for grayscale images
  

def gausspyr_expand(x, sz=None, kernel_a=0.4):
    # Kernel is multipled by 2 to preserve energy when increasing the resolution
    K = 2*np.array([0.25 - kernel_a/2, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2])
  
    # Size of the output image
    if sz is None:
        sz = (x.shape[0]*2, x.shape[1]*2)
    
    x = x.reshape(x.shape[0], x.shape[1], -1) # Add an extra dimension if grayscale
    y = np.zeros([sz[0], sz[1], x.shape[2]]) # Store the result in this array
  
    for cc in range(x.shape[2]): # for each colour channel
        y_a = np.zeros((x.shape[0], sz[1]))
        # Step 1: upsample rows
        y_a[:,::2] = x[:,:,cc]
        # Step 2: filter rows
        y_a = sp.signal.convolve2d(y_a, K.reshape(1,-1), mode='same', boundary='symm')
        # Step 3: upsample columns
        y[::2,:,cc] = y_a
        # Step 4: filter columns
        y[:,:,cc] = sp.signal.convolve2d(y[:,:,cc], K.reshape(-1,1), mode='same', boundary='symm')
    return np.squeeze(y) # remove an extra dimension for grayscale images

class laplacian_pyramid:
    
    @staticmethod
    def decompose(img, levels=-1):    
        """
        Decompose img into a Laplacian pyramid. 
        levels: how many levels should be created (including the base band). When the default (-1) value is used, the maximum possible number of levels is created. 
        """
        # The maximum number of levels
        max_levels = math.floor(math.log2(min(img.shape[0], img.shape[1])))
  
        assert levels < max_levels
  
        if levels == -1: 
            levels = max_levels

        pyramid = []
        gausspyr = []
        # Build Gaussian pyramid
        for i in range(1, levels+1):
            if i == 1:
                gausspyr.append(img)
            else:
                gausspyr.append(gausspyr_reduce(gausspyr[i-2]))

        # Build Laplacian pyramid (Anh Hoàng)
        pyramid.append(gausspyr[0] - gausspyr_expand(gausspyr[1], sz=np.shape(gausspyr[0])))
        for i in range(1, levels):
            pyramid.append(gausspyr[i])
        
        # Buid Laplacian pyramid (original paper)
        # for i in range(0, levels-1):
        #     pyramid.append(gausspyr[i] - gausspyr_expand(gausspyr[i+1], sz=np.shape(gausspyr[i])))
        # pyramid.append(gausspyr[levels-1])

        return pyramid
    
    # Old version of reconstruct
    # @staticmethod
    # def reconstruct(pyramid):    
    #     """
    #     Combine the levels of the Laplacian pyramid to reconstruct an image. 
    #     """
    #     img = None
    #     levels = len(pyramid)
    #     img = pyramid[levels-1]
        
    #     # Perform inverse operation to reconstruct the original image
    #     for i in range(1, levels-1):
    #         # print(levels-i-1)
    #         img = (gausspyr_expand(img, sz=np.shape(pyramid[levels-i-1]))+pyramid[levels-i-1])/2
    #     img = gausspyr_expand(img, sz=np.shape(pyramid[0])) + pyramid[0]
    #     # img = np.maximum(gausspyr_expand(img, sz=np.shape(pyramid[0])), pyramid[0])
    #     return img
    
    @staticmethod
    def reconstruct(pyramid):    
        """
        Combine the levels of the Laplacian pyramid to reconstruct an image.
        Use weighted sum based on sharpness for the last four layers and simple addition for the base layer.
        """
        ### ----------------- Anh Hoàng ----------------- ###
        # Bắt đầu tái tạo từ tầng nhỏ nhất
        reconstructed_image = pyramid[-1]
        # Tính độ sắc nét cho 4 tầng cuối (bỏ qua tầng 0)
        sharpness_scores = [np.sum(np.abs(laplace(layer))) for layer in pyramid[1:]]
        
        # Tái tạo ảnh bằng cách sử dụng trọng số dựa trên độ sắc nét cho 4 tầng cuối
        for i in range(len(pyramid) - 2, 0, -1):
            weight = sharpness_scores[i-1] / sum(sharpness_scores)
            expanded_image = gausspyr_expand(reconstructed_image, sz=np.shape(pyramid[i]))
            reconstructed_image = expanded_image * (1 - weight) + pyramid[i] * weight
        
        # Thêm tầng cơ sở (tầng 0) mà không dùng trọng số
        reconstructed_image = gausspyr_expand(reconstructed_image, sz=np.shape(pyramid[0])) + pyramid[0]
        ### ----------------- Anh Hoàng ----------------- ###

        ### ----------------- Original paper ----------------- ###
        '''
        # Bắt đầu tái tạo từ tầng nhỏ nhất
        reconstructed_image = pyramid[-1]
        # Tính độ sắc nét cho 4 tầng cuối (bỏ qua tầng 0)
        # sharpness_scores = [np.sum(np.abs(laplace(layer))) for layer in pyramid[1:]]
        
        # Tái tạo ảnh bằng cách sử dụng trọng số dựa trên độ sắc nét cho 4 tầng cuối
        for i in range(len(pyramid) - 2, 0, -1):
            # weight = sharpness_scores[i-1] / sum(sharpness_scores)
            expanded_image = gausspyr_expand(reconstructed_image, sz=np.shape(pyramid[i]))
            reconstructed_image = expanded_image + pyramid[i]
        
        # Thêm tầng cơ sở (tầng 0) mà không dùng trọng số
        reconstructed_image = gausspyr_expand(reconstructed_image, sz=np.shape(pyramid[0])) + pyramid[0]
        '''
        ### ----------------- Original paper ----------------- ###

        return reconstructed_image
    
class laplacian_pyramid_original:
    
    @staticmethod
    def decompose(img, levels=-1):    
        """
        Decompose img into a Laplacian pyramid. 
        levels: how many levels should be created (including the base band). When the default (-1) value is used, the maximum possible number of levels is created. 
        """
        # The maximum number of levels
        max_levels = math.floor(math.log2(min(img.shape[0], img.shape[1])))
  
        assert levels < max_levels
  
        if levels == -1: 
            levels = max_levels

        pyramid = []
        gausspyr = []
        # Build Gaussian pyramid
        for i in range(1, levels+1):
            if i == 1:
                gausspyr.append(img)
            else:
                gausspyr.append(gausspyr_reduce(gausspyr[i-2]))
        
        # Buid Laplacian pyramid (original paper)
        for i in range(0, levels-1):
            pyramid.append(gausspyr[i] - gausspyr_expand(gausspyr[i+1], sz=np.shape(gausspyr[i])))
        pyramid.append(gausspyr[levels-1])

        return pyramid
    
    @staticmethod
    def reconstruct(pyramid):    
    
        ### ----------------- Original paper ----------------- ###
        # Bắt đầu tái tạo từ tầng nhỏ nhất
        reconstructed_image = pyramid[-1]
        # Tính độ sắc nét cho 4 tầng cuối (bỏ qua tầng 0)
        # sharpness_scores = [np.sum(np.abs(laplace(layer))) for layer in pyramid[1:]]
        
        # Tái tạo ảnh bằng cách sử dụng trọng số dựa trên độ sắc nét cho 4 tầng cuối
        for i in range(len(pyramid) - 2, 0, -1):
            # weight = sharpness_scores[i-1] / sum(sharpness_scores)
            expanded_image = gausspyr_expand(reconstructed_image, sz=np.shape(pyramid[i]))
            reconstructed_image = expanded_image + pyramid[i]
        
        # Thêm tầng cơ sở (tầng 0) mà không dùng trọng số
        reconstructed_image = gausspyr_expand(reconstructed_image, sz=np.shape(pyramid[0])) + pyramid[0]
        ### ----------------- Original paper ----------------- ###

        return reconstructed_image

class clahe_laplacian_pyramid:
    
    @staticmethod
    def decompose(img, levels=-1):    
        """
        Decompose img into a Laplacian pyramid. 
        levels: how many levels should be created (including the base band). When the default (-1) value is used, the maximum possible number of levels is created. 
        """
        # The maximum number of levels
        max_levels = math.floor(math.log2(min(img.shape[0], img.shape[1])))
  
        assert levels < max_levels
  
        if levels == -1: 
            levels = max_levels

        pyramid = []
        gausspyr = []
        # Build Gaussian pyramid
        adaptive_clahe_img = adaptive_clahe(img)        
        laplacian_sharpened = laplacian_sharpening_dynamic(adaptive_clahe_img)
        img = laplacian_sharpened

        for i in range(1, levels+1):
            if i == 1:
                gausspyr.append(img)
            else:
                gausspyr.append(gausspyr_reduce(gausspyr[i-2]))

        for i in range(0, levels-1):
            pyramid.append(gausspyr[i] - gausspyr_expand(gausspyr[i+1], sz=np.shape(gausspyr[i])))
        pyramid.append(gausspyr[levels-1])

        return pyramid
    
    @staticmethod
    def reconstruct(pyramid):    
        reconstructed_image = pyramid[-1]

        for i in range(len(pyramid) - 2, 0, -1):
            expanded_image = gausspyr_expand(reconstructed_image, sz=np.shape(pyramid[i]))
            reconstructed_image = expanded_image + pyramid[i]
        
        reconstructed_image = gausspyr_expand(reconstructed_image, sz=np.shape(pyramid[0])) + pyramid[0]

        return reconstructed_image

class constrast_pyramid:
    @staticmethod
    # Build def to decompose image using Constrast Pyramid
    def decompose(img, levels=5):
        # Create a Gaussian pyramid
        pyramid = [img]
        for i in range(levels-1):
            img = gausspyr_reduce(img)
            pyramid.append(img)

        # Create a Contrast pyramid
        max_level = pyramid[-1]
        contrast_pyramid = [max_level]
        for i in range(levels-1, 0, -1):
            expanded = gausspyr_expand(pyramid[i], sz=np.shape(pyramid[i-1]))
            # formula: contrast = pyramid[i-1] / expanded - 1
            # contrast = pyramid[i-1] / expanded - 1
            contrast = pyramid[i-1] / expanded
            contrast_pyramid.append(contrast)
        
        # Reverse the contrast pyramid
        contrast_pyramid.reverse()

        return contrast_pyramid

    @staticmethod
    # Build def to reconstruct image using Constrast Pyramid
    def reconstruct(pyramid):
        # Reconstruct the image
        img = pyramid[-1]
        for i in range(len(pyramid)-2, -1, -1):
            expanded = gausspyr_expand(img, pyramid[i].shape)
            # img = expanded * (pyramid[i] + 1)
            img = expanded * (pyramid[i])
        
        return img

class wavelet_pyramid:
    @staticmethod
    def dwt_decompose(image, level=4):
        coeffs = pywt.wavedec2(image, "haar", level=level)
        return coeffs

    @staticmethod
    def dwt_recompose(coeffs):
        return pywt.waverec2(coeffs, "haar")

class dtcwt_pyramid:
    @staticmethod
    def dtcwt_decompose(image, levels=3):
        # Create a DTCWT object
        transform = dtcwt.Transform2d()

        # Perform the DTCWT
        coeffs = transform.forward(image, nlevels=levels)

        return coeffs

    @staticmethod
    def dtcwt_recompose(coeffs):
        # Create a DTCWT object
        transform = dtcwt.Transform2d()

        # Perform the inverse DTCWT
        image = transform.inverse(coeffs)

        return image

class bilateral_filter:
    # Hàm để tính toán các tham số tự động cho bộ lọc Bilateral
    @staticmethod
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
        sigmaColor = contrast * 1.8  # Điều chỉnh giá trị này tùy theo độ tương phản
        sigmaSpace = contrast * 1.5  # Điều chỉnh giá trị này tùy theo độ tương phản

        # Đảm bảo các tham số nằm trong một phạm vi hợp lý
        sigmaColor = max(10, min(sigmaColor, 150))
        sigmaSpace = max(10, min(sigmaSpace, 150))

        return d, sigmaColor, sigmaSpace
    
    @staticmethod
    def blt_decompose(image):
        image = image.astype(np.float32)
        d, sigmaColor, sigmaSpace = bilateral_filter.calculate_bilateral_parameters(image)
        bilateral_filtered = cv2.bilateralFilter(image, 5, sigmaColor, sigmaSpace)
        detail_image = cv2.subtract(image, bilateral_filtered)
        return bilateral_filtered, detail_image
    
    @staticmethod
    def blt_recompose(bilateral_filtered, detail_image):
        return cv2.add(bilateral_filtered, detail_image)

class clahe_filter:
    @staticmethod
    def clahe_decompose(image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Tạo một đối tượng CLAHE với các tham số đã tính toán
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(image)

        # Detail component
        detail_image = cv2.subtract(image, clahe_image)

        return clahe_image, detail_image
    
    @staticmethod
    def clahe_recompose(clahe_image, detail_image):
        if clahe_image.dtype != detail_image.dtype:
            clahe_image = clahe_image.astype(np.float32)
            detail_image = detail_image.astype(np.float32)
        return cv2.add(clahe_image, detail_image)

class adaptive_clahe_filter:
    @staticmethod
    def clahe_decompose(image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        adaptive_clahe_img = adaptive_clahe(image)

        detail_component = cv2.subtract(image, adaptive_clahe_img)

        return adaptive_clahe_img, detail_component

    @staticmethod
    def clahe_recompose(adaptive_clahe_img, detail_component):
        if adaptive_clahe_img.dtype != detail_component.dtype:
            adaptive_clahe_img = adaptive_clahe_img.astype(np.float32)
            detail_component = detail_component.astype(np.float32)
        return cv2.add(adaptive_clahe_img, detail_component)

class nlmd_filter:
    @staticmethod
    def nlmd_decompose(image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if len(image.shape) == 2: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        nlmd_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Convert the denoised image back to grayscale for subtraction
        nlmd_image_gray = cv2.cvtColor(nlmd_image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        nlmd_sub = cv2.subtract(image_gray, nlmd_image_gray)

        return nlmd_image_gray, nlmd_sub

    @staticmethod
    def nlmd_recompose(adaptive_clahe_img, detail_component):
        if adaptive_clahe_img.dtype != detail_component.dtype:
            adaptive_clahe_img = adaptive_clahe_img.astype(np.float32)
            detail_component = detail_component.astype(np.float32)
        return cv2.add(adaptive_clahe_img, detail_component)
    
class multilevel_guided_filter:
    # 1. Hàm thực hiện phép gauss filter
    @staticmethod
    def gauss_filter(image, ksize = 3, sigma = 1):
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # 2. Hàm thực hiện tính toán hình ảnh Gradient Magnitude
    @staticmethod
    def gradient_magnitude(image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)

    # 3. Hàm thực hiện phép Guided Filter với ảnh hướng dẫn I và ảnh đầu vào p, với bán kính r và epsilon
    @staticmethod
    def guided_filter(I, p, r=3, eps=1e-3):
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * I + mean_b

        return q
    
    # 4. Hàm decompose
    @staticmethod
    def decompose(image, levels=3):
        detail_list = []
        base = None

        for i in range(levels):
            # Áp dụng Gauss Filter
            gauss_filtered = multilevel_guided_filter.gauss_filter(image)

            # Tính Gradient Magnitude
            grad_magnitude = multilevel_guided_filter.gradient_magnitude(image)

            # Subtract giữa ảnh gốc và ảnh gradient magnitude
            # Kiểm tra xem image có cần chuyển về kiểu dữ liệu của grad_magnitude không
            if image.dtype != grad_magnitude.dtype:
                image = image.astype(grad_magnitude.dtype)
            grad_diff = cv2.subtract(image, grad_magnitude)

            # Áp dụng Guided Filter
            guided_filtered = multilevel_guided_filter.guided_filter(grad_diff, image)

            # Lưu trữ các ảnh
            sc = cv2.subtract(image, guided_filtered)
            lc = cv2.subtract(guided_filtered, gauss_filtered)

            detail_list.append(sc)
            detail_list.append(lc)

            # Cập nhật ảnh image
            image = gauss_filtered

            # Lưu ảnh cuối cùng
            if i == levels - 1:
                base = gauss_filtered
        
        all_components = [base] + detail_list
        
        return all_components
    
    # 5. Hàm reconstruct
    @staticmethod
    def reconstruct(base, detail_list):
        image = base

        for i in range(len(detail_list)):
            image += detail_list[i]

        return image

def get_energy(img):
    # vec = img.flatten()
    # energy = 0

    # # Tính năng lượng vùng của ảnh đầu vào bằng cách tính tổng bình phương các giá trị pixel
    # for pix in range(0, len(vec)):
    #     energy += vec[pix]**2

    # vec = img.flatten()
    #  # Ma trận trọng số cho từng pixel trong ảnh (các cột 1,2,3 và các hàng 1,2,3): W = 1 / (1+sqrt((i-2)^2+(j-2)^2))
    # W = np.zeros((3,3))
    # for i in range(3):
    #     for j in range(3):
    #         W[i,j] = 1 / (1 + math.sqrt((i-1)**2 + (j-1)**2))
    # # Tính năng lượng vùng của ảnh đầu vào bằng cách tính tổng bình phương của tích chập ảnh với ma trận trọng số
    # energy = np.sum(W * (img**2))

    vec = img.flatten()
    W = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    energy = np.sum(W * (img**2))
    
    return energy

def maximum_region_energy(img_1, img_2, sz_h, sz_w):
     # Khởi tạo ảnh đầu ra với kích thước giống như ảnh đầu vào và đặt giá trị ban đầu bằng 0
    img_output = np.zeros(np.shape(img_1))
    # Kích thước cửa sổ trượt
    h = sz_h
    w = sz_w
    # Tính nửa kích thước cửa sổ để mở rộng ảnh
    ext_x = int(w / 2)
    ext_y = int(h / 2)
    # Mở rộng ảnh đầu vào theo chiều ngang
    edge_x = np.zeros([img_1.shape[0], ext_x], img_1.dtype)
    tem_img_1 = np.hstack((edge_x, img_1, edge_x))
    # Mở rộng ảnh đầu vào theo chiều dọc
    edge_y = np.zeros([ext_y, tem_img_1.shape[1]], img_1.dtype)
    ext_img_1 = np.vstack((edge_y, tem_img_1, edge_y))
    tem_img_2 = np.hstack((edge_x, img_2, edge_x))
    ext_img_2 = np.vstack((edge_y, tem_img_2, edge_y))

    # Duyệt qua từng pixel của ảnh đầu vào
    for y in range(0, img_1.shape[0]):
        for x in range(0, img_1.shape[1]):
            # Lấy cửa sổ trượt tương ứng từ mỗi ảnh mở rộng
            w_1 = ext_img_1[y:y+2*ext_y+1, x:x+2*ext_x+1]
            w_2 = ext_img_2[y:y+2*ext_y+1, x:x+2*ext_x+1]
            # Tính năng lượng vùng cho từng cửa sổ
            RE1 = abs(get_energy(w_1))
            RE2 = abs(get_energy(w_2))
            # Chiến lược ghép ảnh: chọn pixel từ ảnh có năng lượng vùng cao hơn
            if RE1 >= RE2:
                img_output[y, x] = img_1[y, x]
            elif RE1 < RE2:
                img_output[y, x] = img_2[y, x]

    return img_output

def pyramid_fusion_base_level(pyr_1, pyr_2):
    return maximum_region_energy(pyr_1, pyr_2, sz_h=3, sz_w=3)

def save_pyramid(pyramid, output_folder, name):
    os.makedirs(output_folder, exist_ok=True)
    for i, layer in enumerate(pyramid):
        plt.imsave(os.path.join(output_folder, f'{name}{i+1}.png'), layer,cmap='gray', format='png')
