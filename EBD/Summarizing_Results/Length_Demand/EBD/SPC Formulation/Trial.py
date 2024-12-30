def subset_sum(nums, target):
    n = len(nums)
    dp = [[0] * (target + 1) for _ in range(n + 1)]

    # Initialize the first column to True (an empty subset can always achieve the sum of 0)
    for i in range(n + 1):
        dp[i][0] = []

    for i in range(1, n + 1):
        for j in range(1, target + 1):
            # If the current number is greater than the target, exclude it
            if nums[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                # Either include or exclude the current number
                include_subset = dp[i - 1][j - nums[i - 1]]
                if include_subset:
                    dp[i][j] = include_subset + [nums[i - 1]]
                else:
                    dp[i][j] = dp[i - 1][j]

    return dp[n][target]

# Example usage:
numbers = [3, 1, 7, 9, 12]
target_sum = 20

result = subset_sum(numbers, target_sum)
print(result)