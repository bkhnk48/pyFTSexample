#Dưới đây là một đoạn mã Python sử dụng thư viện sklearn để thực hiện phân cụm DBSCAN. Đoạn mã này nhận đầu vào là một mảng các tọa độ trong không gian 2D và in ra số lượng các cụm và mật độ của các cụm đó.
#Lưu ý: Mật độ được tính bằng cách đếm số lượng điểm trong mỗi cụm. Bạn có thể muốn điều chỉnh cách tính mật độ để phù hợp với yêu cầu cụ thể của bạn. Đoạn mã trên giả định rằng bạn đã cài đặt thư viện sklearn. Nếu chưa, bạn có thể cài đặt bằng cách chạy lệnh pip install sklearn. Chúc bạn thành công!
from sklearn.cluster import DBSCAN
import numpy as np

# Giả sử đây là mảng các tọa độ của bạn
coordinates = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# Khởi tạo DBSCAN
db = DBSCAN(eps=3, min_samples=2).fit(coordinates)

# Nhãn của các cụm
labels = db.labels_

# Số lượng cụm (loại trừ nhiễu)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# In số lượng cụm
print('Số lượng cụm: %d' % n_clusters)

# Tính và in mật độ của mỗi cụm
for i in range(n_clusters):
    cluster_points = coordinates[labels == i]
    density = len(cluster_points)
    print('Mật độ của cụm %d: %d' % (i, density))
