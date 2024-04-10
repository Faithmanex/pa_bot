import itertools

# Define the characters and numbers to use
characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
numbers = '123456789'

# Combine characters and numbers
all_chars = characters + numbers

# Generate all combinations of length 18
combinations = itertools.product(all_chars, repeat=18)

# Open a file to write the combinations
with open('combinations.txt', 'w') as file:
    for combination in combinations:
        # Write each combination to the file
        file.write(''.join(combination) + '\n')
