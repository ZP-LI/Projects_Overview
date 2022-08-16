class Student:

    def __init__(self, firtName, lastName, Grades, Bones):
        self.first_name = firtName
        self.last_name = lastName
        self.grades = []
        for g in Grades:
            self.grades.append(float(g))
        self.bonus = Bones
        # print(self.first_name, self.last_name, self.grades, self.bonus, self.passed)
        self.compute_grades()
        self.create_email()
        self.create_student_file()

    def compute_grades(self):
        self.passed = True
        sum_grades = 0.0
        for grade in self.grades:
            if grade > 4.0:
                self.passed = False
                self.avg_grade = 5.0
                return
            else:
                sum_grades += grade

        self.avg_grade = round(sum_grades / len(self.grades), 1)

        if (self.bonus == 'yes') & (self.avg_grade > 1):
            self.avg_grade = round(self.avg_grade - 0.3, 1)
            if self.avg_grade < 1:
                self.avg_grade = 1.0

        return

    def create_email(self):
        self.email = self.first_name[0] + "." + self.last_name + "@tum.de"

        return

    def create_student_file(self):
        file_name = "Grades_" + self.first_name + "_" + self.last_name + ".txt"
        with open(file_name, 'w') as file:
            file.write('{0:30}{1}\n'.format("Name:", self.first_name + " " + self.last_name))
            file.write('{0:30}{1}\n'.format("EMail:", self.email))
            file.write('{0:30}{1}\n'.format("Average Grade:", self.avg_grade))
            file.write("\n")
            if self.passed:
                file.write('{0} {1} passed the course.'.format(self.first_name, self.last_name))
            else:
                file.write('{0} {1} didn\'t pass the course.'.format(self.first_name, self.last_name))


def evaluate_database():
    students = []
    sum_course_grade = 0.0
    num_passed = 0.0
    best ={'name': "", 'grade': 5.0}
    with open("Students.txt", 'r') as file:
        for line in file.readlines():
            student = None
            items = line.split(", ")
            grades = items[1:-1]
            # print(grades)
            student_name = items[0]
            student = Student(student_name.split()[0], student_name.split()[1], grades, items[-1])
            students.append(student)

            sum_course_grade += student.avg_grade
            if student.passed:
                num_passed += 1

            if student.avg_grade < best['grade']:
                best['grade'] = student.avg_grade
                best['name'] = student_name

    print("The average grade of the course was ", round(sum_course_grade / len(students), 2))
    print(round(num_passed / len(students), 2) * 100, "% of the students passed the course")
    print("The best student was ", best['name'], " with an average grade of ", best['grade'])


if __name__ == "__main__":
    evaluate_database()
