from pyscipopt import Model, quicksum

# Tạo mô hình
model = Model("Minimize Objective")

# Thêm biến quyết định
x = model.addVar(name="x", vtype="C")

# Thêm biến trợ giúp để mô hình hóa hàm max
z1 = model.addVar(name="z1", vtype="C")
z2 = model.addVar(name="z2", vtype="C")

# Thêm ràng buộc cho biến trợ giúp
model.addCons(z1 >= x - 6)
model.addCons(z1 >= 0)

model.addCons(z2 >= 4 - x)
model.addCons(z2 >= 0)

# Thiết lập hàm mục tiêu
model.setObjective(z1 + z2, "minimize")

# Giải mô hình
model.optimize()

# In kết quả
solution_x = model.getVal(x)
solution_z1 = model.getVal(z1)
solution_z2 = model.getVal(z2)
print("Giá trị tối ưu của x:", solution_x)
print("Giá trị tối ưu của z1:", solution_z1)
print("Giá trị tối ưu của z2:", solution_z2)
print("Giá trị tối ưu của hàm mục tiêu:", model.getObjVal())
