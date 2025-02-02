import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np 

# Đường dẫn đến file Excel
file_path = './Metric/evaluation_metric_TNO.xlsx'

output_folder = 'charts_tno'
os.makedirs(output_folder, exist_ok=True)

# Danh sách các sheets cần đọc và vẽ biểu đồ
sheets = ['EN', 'MI', 'AG', 'SD', 'CC', 'VIF', 'Qabf']

# Duyệt qua từng sheet và vẽ biểu đồ
for sheet in sheets:
    # Đọc dữ liệu từ sheet hiện tại
    df = pd.read_excel(file_path, sheet_name=sheet, index_col=0)

    # Loại bỏ dòng mean nếu có
    if 'mean' in df.index:
        df = df.drop('mean')

    # Tính giá trị trung bình của mỗi cột và lưu vào một biến
    means = df.mean()

    # Vẽ đồ thị đường
    plt.figure(figsize=(14, 8))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column, marker='o')  # Thêm marker

    plt.title(f'Performance of Fusion Models Across Different Images - {sheet} Sheet')
    plt.xlabel('Image')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Lưu biểu đồ đường vào folder đã tạo
    plt.savefig(f'{output_folder}/{sheet}_performance_line_chart.png')
    plt.close()

    # Tạo một mảng màu ngẫu nhiên cho mỗi cột
    colors = plt.cm.jet(np.linspace(0, 1, len(means)))

    # Vẽ đồ thị cột cho giá trị trung bình, mỗi cột một màu
    plt.figure(figsize=(10, 6))
    means.plot(kind='bar', color=colors, edgecolor='black')
    plt.title(f'Mean Performance of Models - {sheet} Sheet')
    plt.xlabel('Model')
    plt.ylabel('Mean Performance')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Lưu biểu đồ cột vào folder đã tạo
    plt.savefig(f'{output_folder}/{sheet}_mean_performance_chart.png')
    plt.close()

print("All charts have been saved successfully.")