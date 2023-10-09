def permute(nums):
    def backtrack(start):
        if start == len(nums):
            permutations.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]  # Backtrack
    
    permutations = []
    backtrack(0)
    return permutations

def main():
    user_input = input("Enter a list of unique integers separated by spaces: ")
    user_input = user_input.strip().split()
    array = [int(num) for num in user_input]

    result = permute(array)
    print(result)

if __name__ == "__main__":
    main()
