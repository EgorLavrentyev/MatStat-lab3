import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_from_csv(file_name: str):
    data = pd.read_csv(file_name, encoding='1251', sep=';')
    return data

def load_text(filename):
    with open(filename, 'r') as file:
        values = [float(line) for line in file]
    return values

def load_octave(file_name: str):
    a, b, w = load_text(file_name)
    return float(a), float(b), w

def plot_data(data):

    for d in data:
        x = range(1, d.values[:, 0].size + 1)
        y = d.values[:, 0]
        
        for i in range(len(x) - 1):
            plt.plot([x[i], x[i+1]], [y[i], y[i]], color='blue')
    
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.title("Experiment data")
    plt.savefig("image\\data.png")
    plt.show()


def wrapper_plot_interval(data, eps, part, label="Data intervals"):
    for i in range(len(data)):
        data[i].values[:, 1] = data[i].values[:, 0] + eps
        data[i].values[:, 0] -= eps
        plot_interval(data[i].values, part)

    plt.xlabel('n')
    plt.ylabel('mV')
    plt.title(label)
    plt.savefig(f"image\\data_and_intervals{part}.png")
    plt.show()


def plot_interval(data, part):

    x = range(1, data[:, 0].size + 1)
    x2 = np.array(range(1, data[:, 0].size + 1))
    mask = (data[:, 0] < 0.9196246) & (data[:, 1] > 0.919663)
    y_min = np.maximum(data[:, 1], 0.9196246)  # Lower limit for the interval
    y_max = np.minimum(data[:, 0], 0.919663)   # Upper limit for the interval
    
    if part == 2 and np.any(mask):
        plt.vlines(x, data[:, 0], data[:, 1], colors="skyblue")
        plt.vlines(x2[mask], y_min[mask], y_max[mask], colors="orchid")
    elif part == 3:
        plt.vlines(x, data[:, 0], data[:, 1], colors="skyblue")
        plt.hlines(0.919603, 0, 200, colors="orchid", lw=2)
    else:
        plt.vlines(x, data[:, 0], data[:, 1], colors="skyblue")




def find_mu(data, interval, eps):
    count = 0
    for i in range(len(data)):
        if ((data[i].values[:, 0] + eps) > interval[1]).any() and ((data[i].values[:, 0] - eps) < interval[0]).any():
            count += 1
    return count


def plot_mu(data, eps):
    df = data[0]
    lb = [df.loc[i][0] - eps for i in range(len(df))]
    rb = [df.loc[i][0] + eps for i in range(len(df))]
    borders = sorted(lb + rb)
    freq = [0 for _ in range(2 * len(df) - 1)]
    for i in range(2 * len(df) - 1):
        for j in range(len(df)):
            if (df.loc[j][0] - eps <= borders[i]) and (df.loc[j][0] + eps >= borders[i + 1]):
                freq[i] += 1

    moda_val = max(freq)
    moda_indexes = [i for i in range(len(freq)) if freq[i] == moda_val]
    moda = [borders[moda_indexes[0]], borders[moda_indexes[-1]]]

    print('max mu i', moda_val)
    print('moda indexes', moda_indexes)
    print('moda', moda)

    plt.axhline(y=moda_val, color='orchid', linestyle='--', label='moda')
    plt.plot(borders[:-1], freq, label='1')
    plt.legend()
    plt.xlabel('mV')
    plt.ylabel('mu')
    plt.show()


if __name__ == "__main__":
    data = []
    data.append(load_from_csv("Chanel_1_400nm_2mm.csv"))
    plot_data(data)

    eps = 1.5 * 1e-4
    size_range = range(len(data))
    wrapper_plot_interval(data, eps, 1)
    wrapper_plot_interval(data, eps, 2)
    wrapper_plot_interval(data, eps * 7.7490, 3)

    eps = 2.5 * 1e-4
    plot_mu(data, eps)

