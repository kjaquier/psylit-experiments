from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display, Markdown

from models.cascades import MultiCascades


def plot_dt(multicasc: MultiCascades, log=False):
    dt = multicasc.get_dt()
    display(Markdown("Most common values:"))
    display(dt.value_counts().nlargest(10))
    dt = dt.dropna()
    if log:
        plt.yscale('log')
    r = np.arange(1e-10, dt.max(), 1)
    plt.xticks(r)
    plt.xlabel('dt')
    plt.ylabel('Log Frequency')
    plt.title('Log Distribution of dt')
    sns.distplot(dt, kde=False, label='dt', bins=np.arange(dt.max()))
