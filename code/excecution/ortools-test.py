from ortools.sat.python import cp_model  # pip install -U ortools

# Specifying the input
weights = [395, 658, 113, 185, 336, 494, 294, 295, 256, 530, 311, 321, 602, 855, 209, 647, 520, 387, 743, 26, 54, 420, 667, 971, 171, 354, 962, 454, 589, 131, 342, 449, 648, 14, 201, 150, 602, 831, 941, 747, 444, 982, 732, 350, 683, 279, 667, 400, 441, 786, 309, 887, 189, 119, 209, 532, 461, 420, 14, 788, 691, 510, 961, 528, 538, 476, 49, 404, 761, 435, 729, 245, 204, 401, 347, 674, 75, 40, 882, 520, 692, 104, 512, 97, 713, 779, 224, 357, 193, 431, 442, 816, 920, 28, 143, 388, 23, 374, 905, 942]
values = [71, 15, 100, 37, 77, 28, 71, 30, 40, 22, 28, 39, 43, 61, 57, 100, 28, 47, 32, 66, 79, 70, 86, 86, 22, 57, 29, 38, 83, 73, 91, 54, 61, 63, 45, 30, 51, 5, 83, 18, 72, 89, 27, 66, 43, 64, 22, 23, 22, 72, 10, 29, 59, 45, 65, 38, 22, 68, 23, 13, 45, 34, 63, 34, 38, 30, 82, 33, 64, 100, 26, 50, 66, 40, 85, 71, 54, 25, 100, 74, 96, 62, 58, 21, 35, 36, 91, 7, 19, 32, 77, 70, 23, 43, 78, 98, 30, 12, 76, 38]
capacity = 2000

# Now we solve the problem
model = cp_model.CpModel()
xs = [model.new_bool_var(f"x_{i}") for i in range(len(weights))]

model.add(sum(x * w for x, w in zip(xs, weights)) <= capacity)
model.maximize(sum(x * v for x, v in zip(xs, values)))

solver = cp_model.CpSolver()
solver.solve(model)

print("Optimal selection:", [i for i, x in enumerate(xs) if solver.value(x)])
print("Total packed value:", solver.objective_value)