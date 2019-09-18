import plotly.express as px
import scipy.integrate as integrate
import pandas as pd
import seaborn as sns
import numpy as np

class ECDF:
    
    def __init__(self, x, sample=2**15+1):
        n = x.size
        thres = sample / n
        s = [0,0]
        while len(s) % 2 == 0:
            s = np.random.sample(n) < thres
        print(len(s[s]), 'samples')
        y = np.arange(n) / n
        xs = np.sort(x[s])
        ns = xs.size
        ys = np.arange(ns) / ns
        
        self.support = x.min(), x.max()
        self.x = x
        self.xs = xs
        self.ys = ys
        self.step = StepFunction(np.sort(x), y)
    
    def expected(self, support=None):
        a, b = support or self.support
        print("Support:", a, b)
        cdf = self.step
        integ = integrate.romb(cdf(self.ys))
        return b - integ
        
    def plot_cdf(self, sample=100_000, xlabel='x', ylabel='ECDF', title=None):
        title = title or f"ECDF of {xlabel}"
        cdf = self.step
        df = pd.DataFrame({'x': self.xs, 'y': cdf(self.xs)})
        return px.line(df, x='x', y='y', labels=dict(x=xlabel, y='ECDF'), render_mode='webgl', title=title, template='seaborn')
        
    def plot_log_pdf(self, sample=100_000, xlabel='x', ylabel='ECDF', title=None):
        title = title or f"ECDF of {xlabel}"
        cdf = self.step
        df = pd.DataFrame({'x': self.xs[self.xs>0]})
        return px.histogram(df, x='x', histnorm='probability density', log_x=True, nbins=1000)
    
    def plot_log_ecdf(self, sample=100_000, xlabel='x', ylabel='ECDF', title=None):
        title = title or f"ECDF of {xlabel}"
        cdf = self.step
        df = pd.DataFrame({'x': self.xs[self.xs>0]})
        return px.histogram(df, x='x', histnorm='probability density', log_x=True, nbins=1000, cumulative=True)