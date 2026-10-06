import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(3, 1, sharex=True)

    def plot(self, index, x, y, label=None, linestyle="-", color="blue"):
        self.ax[index].plot(x, y, label=label, linestyle=linestyle, color=color)

    def show(self):
        for ax in self.ax:
            ax.legend()
            ax.grid()

        self.ax[2].set_xlabel("Time [s]")
        plt.show(block=False)

        try:
            while plt.get_fignums():
                plt.pause(0.1)
        except KeyboardInterrupt:
            plt.close("all")



