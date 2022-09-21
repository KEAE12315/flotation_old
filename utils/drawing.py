import matplotlib.pyplot as plt


def pltxys(x, ys):
    for k, v in x.items():
        x = v

    for k, v in ys.items():
        plt.plot(x, v, label=k)

    plt.legend()
