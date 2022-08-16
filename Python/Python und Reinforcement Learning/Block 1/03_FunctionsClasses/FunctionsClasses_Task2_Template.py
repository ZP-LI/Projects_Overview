### Enter your code here ###
def square_numbers(n):
    assert n >= 0 & n % 1 == 0, "input number is not an natural number"
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 2 ** 2
    else:
        return 3 * square_numbers(n - 1) - 3 * square_numbers(n - 2) + square_numbers(n - 3)


### End of your code ###


if __name__ == "__main__":
    for i in range(10):
        print('Number:\t', i, '\t\tSquared number:\t', square_numbers(i))





