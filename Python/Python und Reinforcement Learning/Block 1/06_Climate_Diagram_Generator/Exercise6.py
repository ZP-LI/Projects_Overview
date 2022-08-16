import numpy as np
import matplotlib.pyplot as plt

LOC_NAME = 'Zugspitze'


def read_text():
    f_name = 'Database_' + LOC_NAME + '.txt'

    date = []

    with open(f_name, 'r') as file:
        content = file.readlines()
        num_rows = len(content) - 3
        num_columns = len(content[4].split())
        data = np.zeros((num_rows, num_columns))
        for i in range(num_rows):
            data[i, :] = content[i + 3].split()
            date.insert(0, str(int(data[i, 1])))

    T_avg = data[:, 5].reshape(-1)[::-1]
    sun_time = data[:, -4].reshape(-1)[::-1]
    rainfall = data[:, -2].reshape(-1)[::-1]

    res = (date, T_avg, sun_time, rainfall)

    return res


def evaluate_sun_data(data):
    sun_time = data[2]
    sum_sun_time = np.sum(sun_time)
    avg_time = np.average(sun_time)
    max_idx = np.argmax(sun_time)
    print(data[0])
    max_day = data[0][max_idx]
    print('In total there were ', round(sum_sun_time, 1), ' hours of sun in ', LOC_NAME, ', which on average were ', round(avg_time, 2), ' hours per day.')
    print('The sunniest day was on ', max_day[0:4], '/', max_day[4:6], '/', max_day[6:8], ' with ', sun_time[max_idx], ' hours of sun.')


def calculate_month(data):
    T_sum_mon = np.zeros(12)
    rainfall_mon = np.zeros(12)
    Days = np.zeros(12)
    for i in range(12):
        for j in range(len(data[0])):
            if int(data[0][j][4:6]) == (i + 1):
                Days[i] = Days[i] + 1
                T_sum_mon[i] = T_sum_mon[i] + float(data[1][j])
                rainfall_mon[i] = rainfall_mon[i] + float(data[3][j])
    T_avg_mon = T_sum_mon / Days

    return T_avg_mon, rainfall_mon


def plot_climate_diagram(temp, rain):
    month = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    ax1 = plt.subplot()
    plt.plot(temp, color='r')
    plt.ylabel("Temperature/°C", fontdict={'color': 'red'})
    plt.yticks([0, 20, 40, 60, 80, 100])
    plt.xlabel("Month")

    ax2 = ax1.twinx()
    ax2.plot(rain, color='b')
    plt.ylabel("Rainfall/mm", fontdict={'color': 'blue'})
    plt.yticks([0, 50, 100, 150, 200])

    plt.title('Climate Diagram of ' + LOC_NAME, fontdict={'size': 16}, fontweight='bold')
    plt.xticks(range(12), month)

    plt.savefig('Climate_Diagram_' + LOC_NAME + '.png')
    plt.show()


if __name__ == "__main__":
    data = read_text()
    evaluate_sun_data(data)
    temp, rain = calculate_month(data)
    plot_climate_diagram(temp, rain)
