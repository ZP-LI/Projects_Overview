import numpy as np


class Circle:

    radius = 5

    def __int__(self):
        return

    def __init__(self, Radius):
        self.radius = Radius

    def compute_area(self):
        return np.pi * self.radius ** 2

    def compute_perimeter(self):
        return 2 * np.pi * self.radius


class Rectangle:

    width = 5
    length = 5

    def __int__(self):
        return

    def __init__(self, Width, Length):
        self.width = Width
        self.length = Length

    def compute_area(self):
        return self.width * self.length

    def compute_perimeter(self):
        return 2 * (self.width + self.length)


def generate_objects(number):
    obj_list = np.random.randint(1, 3, 10)
    objects = []
    for obj in obj_list:
        if obj == 1:  # Generate a circle
            r = np.random.randint(1, 11)
            object = Circle(r)
            objects.append(object)
        else:  # Generate a rectangle
            w = np.random.randint(1, 11)
            l = np.random.randint(1, 11)
            object = Rectangle(w, l)
            objects.append(object)

    return objects


if __name__ == '__main__':
    number = 10
    objects = generate_objects(number)
    sum_area = 0
    sum_perimeter = 0
    for object in objects:
        sum_area += object.compute_area()
        sum_perimeter += object.compute_perimeter()

    print('mean area = ', sum_area/number, 'and sum of the perimeters = ', sum_perimeter)