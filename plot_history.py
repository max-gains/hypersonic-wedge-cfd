import matplotlib.pyplot as plt
import pandas as pd
import sys

def main(hist_name):
    df = pd.read_csv(hist_name, skipinitialspace=True)
    df = df.rename(columns=lambda x: x.strip())

    # cmap = plt.get_cmap('viridis')
    # colors = [cmap(i / 3) for i in range(3)]
    colors = ['k', 'k', 'k']

    # Plot residuals
    if 'bgs[Rho][0]' in df.columns: # primal
        plt.plot(df['bgs[Rho][0]'], label='bgs[Rho][0]', color=colors[0], linestyle='-')
        plt.plot(df['bgs[RhoU][0]'], label='bgs[RhoU][0]', color=colors[0], linestyle='--')
        plt.plot(df['bgs[T][1]'], label='bgs[T][1]', color=colors[0], linestyle=':')
    else:
        raise Exception('Bad history file "{}" specified.'.format(hist_name))

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('N iters')
    plt.ylabel('log RMS')
    # plt.ylim([-10, -2])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_history.py <history_file_1.csv> <history_file_2.csv> ...")
        sys.exit(1)
    for hist in sys.argv[1:]: main(hist)

