import numpy as np


def task1(database):
    ### Enter your code here ###
    database[database <= 0] = 0

    return database
    ### End of your code ###

def task2(number):
    pass
    ### Enter your code here ###
    if number <= 3:
        return number > 1
    if number % 2 == 0 or number % 3 == 0:
        return False
    i = 5
    while i ** 2 <= number:
        if number % i == 0 or number % (i + 2) == 0:
            return False
        i += 6
    return True
    ### End of your code ###

def task3():
    numbers = np.random.randint(0, 3, size=3)
    print(numbers)

    ### Enter your code here ###
    if numbers[0] == numbers[1] & numbers[0] == numbers[2] & numbers[1] == numbers[2]:
        print('All numbers are the same')
    elif numbers[0] == numbers[2]:
        print('The first and the last number are the same')
    elif numbers[0] == numbers[1]:
        print('The first two numbers are the same')
    elif numbers[1] == numbers[2]:
        print('The last two numbers are the same')
    else:
        print('All numbers are different')
    ### End of your code ###


if __name__ == "__main__":
    print('Task 1:')
    database = np.array([-1, 2, 6, -5, 0, -3, 8, 0.01, 5, -3, 7, 8, 5, 3, 2, 0, -0.11, 4.45])
    print(task1(database))
    print('\n\nTask 2:')
    numbers2check = (9, 5, 10, 17, 563, 1998, 1973)
    print(numbers2check)
    result = []
    for number in numbers2check:
        result.append(task2(number))
    print(result)
    print('\n\nTask 3:')
    task3()

