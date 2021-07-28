import matplotlib.pyplot as plt
import seaborn as sns


def plot_series(timeseries):
    """
    handy plots for a timeseries, histogram and also values over time
    
    Args:
        timeseries (pd.Series) : data to plot
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    sns.histplot(timeseries, kde=True, ax=ax[0])
    ax[0].set_title(f"{timeseries.name} Histogram")

    ax[1].plot(timeseries)
    ax[1].set_title(f"{timeseries.name} Over Time")
    plt.show()
