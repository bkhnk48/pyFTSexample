from pyscipopt import Model, quicksum

model = Model("Simple linear optimization")
vars_by_index_i = {}
vars_by_index_j = {}
vars_and_costs = {}
# Khởi tạo tập hợp để theo dõi các điểm i và j cần loại trừ
exclude_i = set()
exclude_j = set()

# Đọc file và tạo biến quyết định
with open('simpleInput2.txt', 'r') as file:
	for line in file:
		parts = line.strip().split()
		if parts[0] == 'a':
			i, j, cij = parts[1], parts[2], int(parts[5])
			model.addVar(vtype="B", name=f"x{i}_{j}")
			vars_by_index_i.setdefault(i, []).append(f"x{i}_{j}")
			vars_by_index_j.setdefault(j, []).append(f"x{i}_{j}")
			vars_and_costs[f"x{i}_{j}"] = cij
		elif parts[0] == 'n':
			# Lưu trữ chỉ số i hoặc j để sau này thêm ràng buộc
			index = parts[1]
			if parts[2] == '1':
				exclude_i.add(parts[1])
				vars_by_index_i.setdefault(index, [])
			elif parts[2] == '-1':
				exclude_j.add(parts[1])
				vars_by_index_j.setdefault(index, [])

# Retrieve all variables from the model
all_vars = model.getVars()
# Create a dictionary to map variable names to variable objects
var_dict = {v.name: v for v in all_vars}
#tất cả các cung có điểm nguồn là điểm xuất phát của một AGV thì chúng có tổng bằng 1
for i, var_names in vars_by_index_i.items():
	if i in exclude_i:
		#model.addCons(quicksum(model.getVarByName(name) for name in var_names) == 1)
		model.addCons(quicksum(var_dict[name] for name in var_names if name in var_dict) == 1)
	
# Thêm ràng buộc: tổng tất cả các xji = 1 với mỗi j có giá trị '-1'
for j, var_names in vars_by_index_j.items():
	if j in exclude_j:
		#model.addCons(quicksum(model.getVarByName(name) for name in var_names) == 1)
		model.addCons(quicksum(var_dict[name] for name in var_names if name in var_dict) == 1)
	
# Thêm ràng buộc: tổng tất cả các xij = tổng tất cả các xjk cho mỗi j
for j in vars_by_index_j.keys():
	if j in vars_by_index_i and j not in exclude_i and j not in exclude_j:
		#model.addCons(quicksum(model.getVarByName(name) for name in vars_by_index_i[j]) == quicksum(model.getVarByName(name) for name in vars_by_index_j[j]))
		sum_i = quicksum(var_dict[name] for name in vars_by_index_i[j] if name in var_dict)
		sum_j = quicksum(var_dict[name] for name in vars_by_index_j[j] if name in var_dict)
		model.addCons(sum_i == sum_j)

#model.setObjective(quicksum(vars_and_costs[var_name] * model.getVarByName(var_name) for var_name in vars_and_costs), "minimize")
model.setObjective(quicksum(vars_and_costs[var_name] * var_dict[var_name] for var_name in vars_and_costs), "minimize")

model.optimize()
if model.getStatus() == "optimal":
	print("Optimal value:", model.getObjVal())
	print("Solution:")
	# Lấy tất cả các biến từ mô hình
	vars = model.getVars()
	# In giá trị của tất cả các biến
	for var in vars:
		if model.getVal(var) > 0:
			print(f"{var.name} = {model.getVal(var)}")
else:
	print("Không tìm thấy giải pháp tối ưu.")

