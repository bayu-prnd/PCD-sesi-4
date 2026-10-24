import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# 1. Membaca citra awal
# Pastikan Anda memiliki file 'gambar.jpg' di folder yang sama
img_rgb = iio.imread('gambar.jpg')
img_rgb = img_rgb[::4, ::4]

# 2. Konversi citra ke Grayscale menggunakan Numpy (Rumus Luminansi)
img_gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])

# 3. Peningkatan Citra menggunakan Scipy NDImage
# a. Noise Reduction: Menghaluskan citra dengan Gaussian Filter
img_smoothed = ndi.gaussian_filter(img_gray, sigma=2.0)

# b. Feature Extraction: Mendeteksi tepi objek (Sobel Filter)
sobel_x = ndi.sobel(img_smoothed, axis=0)
sobel_y = ndi.sobel(img_smoothed, axis=1)
# Menggabungkan gradien x dan y untuk mendapatkan garis tepi yang jelas
img_edges = np.hypot(sobel_x, sobel_y)

# 4. Visualisasi hasil (untuk keperluan Screenshot)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title('1. Citra Awal (Grayscale)')
axes[0].axis('off')

axes[1].imshow(img_smoothed, cmap='gray')
axes[1].set_title('2. Ditingkatkan (Gaussian Blur)')
axes[1].axis('off')

axes[2].imshow(img_edges, cmap='gray')
axes[2].set_title('3. Analisis Tepi (Sobel Filter)')
axes[2].axis('off')

plt.tight_layout()
plt.show()