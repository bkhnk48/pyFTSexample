import os  # Import thư viện os để tương tác với hệ điều hành
import sys  # Import thư viện sys để tương tác với Python runtime environment
import pandas as pd  # Import thư viện pandas để xử lý dữ liệu dạng bảng
import numpy as np  # Import thư viện numpy để xử lý dữ liệu dạng mảng nhiều chiều

import plotly.express as px  # Import thư viện plotly.express để vẽ đồ thị
import plotly.graph_objects as go  # Import thư viện plotly.graph_objects để tạo đồ thị tương tác
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # Import hàm plot_acf và plot_pacf từ thư viện statsmodels để vẽ đồ thị tự tương quan và tương quan riêng phần
import matplotlib.pylab as plt  # Import thư viện matplotlib.pylab để vẽ đồ thị
from statsmodels.api import tsa  # Import thư viện statsmodels.api để xử lý dữ liệu chuỗi thời gian

#pylab inline  # Chạy pylab inline để hiển thị đồ thị ngay trong notebook

from pyFTS.partitioners import Grid  # Import class Grid từ thư viện pyFTS.partitioners để phân chia dữ liệu
from pyFTS.models import chen, cheng  # Import các mô hình chen và cheng từ thư viện pyFTS.models
from pyFTS.common import Util , Transformations  # Import các hàm Util và Transformations từ thư viện pyFTS.common
from pyFTS.benchmarks import Measures  # Import thư viện Measures từ pyFTS.benchmarks để đánh giá mô hình

from sklearn.metrics import mean_squared_error, mean_absolute_error  # Import các hàm mean_squared_error và mean_absolute_error từ thư viện sklearn.metrics để đánh giá mô hình

raw_df = pd.read_csv('online_retail_II.csv', sep=',')  # Đọc dữ liệu từ file csv
raw_df.head()  # Hiển thị 5 dòng đầu tiên của DataFrame

raw_df.describe()  # Mô tả thống kê chung của DataFrame

raw_df.describe(include=['O'])  # Mô tả thống kê chung của các cột dữ liệu dạng object trong DataFrame

raw_df.info()  # Hiển thị thông tin tổng quan của DataFrame

cancellation_dataset = raw_df.loc[raw_df['Invoice'].str.contains("C", regex=False, na=False)]  # Lọc dữ liệu hủy bỏ từ DataFrame
print(cancellation_dataset.sample(15))  # In ra 15 mẫu ngẫu nhiên từ tập dữ liệu hủy bỏ

idx_tmp = cancellation_dataset.index  # Lấy index của tập dữ liệu hủy bỏ
raw_df = raw_df.drop(idx_tmp)  # Xóa các dòng tương ứng với tập dữ liệu hủy bỏ từ DataFrame gốc
raw_df = raw_df.drop(raw_df.loc[raw_df.Quantity<0].index)  # Xóa các dòng có 'Quantity' nhỏ hơn 0 từ DataFrame
raw_df.shape  # Hiển thị kích thước của DataFrame

input_df = raw_df[['InvoiceDate', 'Quantity']]  # Tạo DataFrame mới với 'InvoiceDate' và 'Quantity'
input_df.head()  # Hiển thị 5 dòng đầu tiên của DataFrame mới

# Chuyển đổi 'InvoiceDate' thành định dạng datetime
raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'])

# Tạo DataFrame mới với 'InvoiceDate' và 'Quantity'
input_df = raw_df[['InvoiceDate', 'Quantity']]

# Đặt 'InvoiceDate' làm index cho DataFrame
input_df = input_df.set_index('InvoiceDate')

# Nhóm dữ liệu theo ngày và tính tổng 'Quantity'
input_df = input_df.groupby(pd.Grouper(freq='D')).sum()
input_df.head()  # Hiển thị 5 dòng đầu tiên của DataFrame sau khi nhóm

input_df.shape  # Hiển thị kích thước của DataFrame
px.line(input_df, x=input_df.index, y="Quantity")  # Vẽ đồ thị dạng đường cho 'Quantity' theo thời gian
print(plot_acf(input_df))  # In đồ thị tự tương quan của 'Quantity'
print(plot_pacf(input_df))  # In đồ thị tương quan riêng phần của 'Quantity'

composition = tsa.seasonal_decompose(input_df)  # Phân rã chuỗi thời gian thành xu hướng, chu kỳ và dư thừa
print(composition.plot())  # In đồ thị phân rã

data = input_df.Quantity.values  # Lấy giá trị của 'Quantity' từ DataFrame
tdiff = Transformations.Differential(1)  # Khởi tạo biến đổi Differential với độ trễ là 1

# boxcox = Transformations.BoxCox(0)

# diff_data = tdiff.apply(data)

train = data[:-30]  # Tạo tập huấn luyện bằng cách lấy tất cả dữ liệu trừ 30 ngày cuối
test = data[-30:]  # Tạo tập kiểm tra bằng cách lấy 30 ngày cuối của dữ liệu

fs = Grid.GridPartitioner(data=data,npart=20, transformation=tdiff)  # Khởi tạo GridPartitioner với 20 phần và biến đổi Differential

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[25,10])  # Khởi tạo đồ thị

fs.plot(ax)  # Vẽ đồ thị phân chia
model = cheng.TrendWeightedFTS(partitioner=fs, transformation=tdiff)  # Khởi tạo mô hình TrendWeightedFTS với GridPartitioner và biến đổi Differential
model.fit(train)  # Huấn luyện mô hình với tập huấn luyện
model.append_transformation(tdiff)  # Thêm biến đổi Differential vào mô hình
print(model)  # In thông tin mô hình
prediction = model.predict(test, transformation=tdiff)  # Dự đoán tập kiểm tra với mô hình

fig = go.Figure()  # Khởi tạo đồ thị
fig.add_trace(go.Scatter(x=input_df.index[-30:], y=test,
                    mode='lines',
                    name='Real'))  # Thêm đường thực tế vào đồ thị
fig.add_trace(go.Scatter(x=input_df.index[-30:], y=prediction,
                    mode='lines',
                    name='Prediction'))  # Thêm đường dự đoán vào đồ thị
#print("=====================CREATE====================")
#fig.show()  # Hiển thị đồ thị
fig.write_image("fig.png")  # Lưu đồ thị dưới dạng hình ảnh
#print("======================END====================")
