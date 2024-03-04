from itertools import combinations

my_list = [[1,2],[3],[4,5]]
my_list_2=[[4,5],[3],[1,2]]

# Find all possible 2-element combinations
combinations_list = list(combinations(my_list, 2))
combinations_list_2 = list(combinations(my_list_2, 2))

# for combo in combinations_list:
#     a=combo[0]
#     b=combo[1]
#     print(a)
#     print(b)


# Print the result
print(combinations_list)
print(combinations_list_2)


