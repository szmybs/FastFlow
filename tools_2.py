from PIL import Image
import numpy as np


def count_connected_white_regions(image_path):
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)
    white_threshold = 127
    binary_data = np.where(img_data > white_threshold, 1, 0)

    labeled, num_features = ndimage.label(binary_data)
    region_sizes = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        region_sizes.append(size)

    return region_sizes

if __name__ == "__main__":
    from scipy import ndimage
    import os
    
    subgraph_size = []
    folder_path = "./data/plate/ground_truth/"
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取图片文件的完整路径
            file_path = os.path.join(root, file)

            # 仅处理图像文件（例如png, jpg, jpeg等）
            if file.lower().endswith(('_mask.png', '_mask.jpg', '_mask.jpeg', '_mask.bmp', '_mask.tiff')):   
                print(file_path) 
                sizes = count_connected_white_regions(file_path)
                sizes = np.array(sizes)
                sizes = sizes[sizes >= 10]
                subgraph_size.append(np.max(sizes))
    subgraph_size_np = np.array(subgraph_size)
    subgraph_size_np = subgraph_size_np[subgraph_size_np >= 10]
    q = np.percentile(subgraph_size_np, 1)  
    print(q)
    print()