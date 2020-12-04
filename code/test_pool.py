from multiprocessing import Pool

def f(x):
    return x * x

if __name__ == '__main__':
    x = range(1, 21)
    chunck = []
    for i in range(0, len(x), 20 / 4):
        chunck.append(x[i: i + 20 / 4])
    print(chunck)