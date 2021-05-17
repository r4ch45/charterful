```python
# OPTIONAL: Load the "autoreload" extension so that code can change
%load_ext autoreload

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
%autoreload 2

%load_ext lab_black

import sys

sys.path.append("..")
from src.data import make_dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import ppscore as pps
import scipy
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    The lab_black extension is already loaded. To reload it, use:
      %reload_ext lab_black
    

# What are Gas Fired Power Stations?

[https://www.energy.gov/fe/how-gas-turbine-power-plants-work]

Gas Fired Power Stations work using gas turbine engines, turbine engines generally have four main parts to the process:

1. **Suck** - Air is drawn in to the engine

2. **Squeeze** - This air is pressurised, and fed in to the combustion chamber at a very high speed.

3. **Bang** - The combustion system feeds in fuel to mix with the air, this is then ignited at high temperatures.

4. **Blow** - The high temperatures expands the gas out the back of the engine, turning rotar blades which are dual prupose. Firstly, the blades draw more gas in to the system, and secondly they are used to generate electricity.

There are two types of gas fired power stations primarily used in the UK.

The first is CCGT (Combined Cycle Gas Turbine), this involves using a gas turbine to drive a gas turbine generator. Any excess heat is recovered by producing steam which drives a steam turbine generator for more electricity.

The second is OCGT (Open Cycle Gas Turbine), this is similar to CCGT but has an initial step of compressing the gas before being injected in the combustor.

# Historically


```python
volume = make_dataset.prepare_gas_data(
    r"C:\\Users\\rachel.hassall\\git\\charterful\\data\\raw\\GAS_VOLUME.csv"
)

volume["YEAR"] = volume["GAS_DAY"].dt.year

volume.groupby(["YEAR"])["ITEM"].nunique()
```




    YEAR
    2016    187
    2017    186
    2018    185
    2019    186
    2020    183
    Name: ITEM, dtype: int64



There were 36 power stations drawing gas from the NTS in 2020.

# Balancing Renewables

Gas electricity generation plays a crucial role in balancing renewable electricity generation. Looking at Wind Generation from direct connections to the Electricity System (excluding distributed generation) and CCGT generation average values per gas day, as a fraction of total electricity generation (to counteract seasonal variations) it's visually clear that power station generation is used to counteract variation in wind.


```python
elec
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>CCGT</th>
      <th>OIL</th>
      <th>COAL</th>
      <th>NUCLEAR</th>
      <th>WIND</th>
      <th>PS</th>
      <th>NPSHYD</th>
      <th>OCGT</th>
      <th>OTHER</th>
      <th>INTFR</th>
      <th>INTIRL</th>
      <th>INTNED</th>
      <th>INTEW</th>
      <th>BIOMASS</th>
      <th>INTNEM</th>
      <th>INTELEC</th>
      <th>INTIFA2</th>
      <th>INTNSL</th>
      <th>TED</th>
    </tr>
    <tr>
      <th>GAS_DAY</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-12-31</th>
      <td>4.0</td>
      <td>6739.888889</td>
      <td>0.0</td>
      <td>3462.444444</td>
      <td>8274.888889</td>
      <td>3208.000000</td>
      <td>4.222222</td>
      <td>711.000000</td>
      <td>0.000000</td>
      <td>2019.222222</td>
      <td>1756.000000</td>
      <td>14.000000</td>
      <td>877.333333</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>27099.000000</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>32.5</td>
      <td>7374.125000</td>
      <td>0.0</td>
      <td>3534.270833</td>
      <td>8160.187500</td>
      <td>4092.708333</td>
      <td>306.958333</td>
      <td>806.666667</td>
      <td>0.000000</td>
      <td>1984.187500</td>
      <td>1735.208333</td>
      <td>48.541667</td>
      <td>899.916667</td>
      <td>50.375000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29025.645833</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>80.5</td>
      <td>9566.541667</td>
      <td>0.0</td>
      <td>3520.083333</td>
      <td>7785.083333</td>
      <td>4282.833333</td>
      <td>330.750000</td>
      <td>847.708333</td>
      <td>0.000000</td>
      <td>1951.354167</td>
      <td>1783.333333</td>
      <td>8.000000</td>
      <td>879.541667</td>
      <td>7.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31043.062500</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>128.5</td>
      <td>10650.979167</td>
      <td>0.0</td>
      <td>3436.666667</td>
      <td>7821.333333</td>
      <td>3811.958333</td>
      <td>300.500000</td>
      <td>721.812500</td>
      <td>0.000000</td>
      <td>2020.916667</td>
      <td>1801.750000</td>
      <td>87.291667</td>
      <td>891.375000</td>
      <td>100.791667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31773.875000</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>176.5</td>
      <td>14029.125000</td>
      <td>0.0</td>
      <td>5758.833333</td>
      <td>7735.750000</td>
      <td>3672.875000</td>
      <td>326.458333</td>
      <td>790.145833</td>
      <td>4.375000</td>
      <td>2038.208333</td>
      <td>1946.291667</td>
      <td>48.208333</td>
      <td>846.958333</td>
      <td>34.458333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37408.187500</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-02-19</th>
      <td>90058.5</td>
      <td>7463.062500</td>
      <td>0.0</td>
      <td>680.687500</td>
      <td>4836.541667</td>
      <td>10810.520833</td>
      <td>273.083333</td>
      <td>727.354167</td>
      <td>10.979167</td>
      <td>181.750000</td>
      <td>1422.500000</td>
      <td>133.500000</td>
      <td>752.083333</td>
      <td>170.833333</td>
      <td>1857.583333</td>
      <td>728.625000</td>
      <td>0.0</td>
      <td>572.041667</td>
      <td>0.0</td>
      <td>120679.645833</td>
    </tr>
    <tr>
      <th>2021-02-20</th>
      <td>90106.5</td>
      <td>6036.666667</td>
      <td>0.0</td>
      <td>249.708333</td>
      <td>4362.895833</td>
      <td>10582.770833</td>
      <td>194.958333</td>
      <td>722.562500</td>
      <td>3.666667</td>
      <td>183.229167</td>
      <td>1155.333333</td>
      <td>182.583333</td>
      <td>537.958333</td>
      <td>277.333333</td>
      <td>1142.520833</td>
      <td>696.500000</td>
      <td>0.0</td>
      <td>620.958333</td>
      <td>0.0</td>
      <td>117056.145833</td>
    </tr>
    <tr>
      <th>2021-02-21</th>
      <td>90154.5</td>
      <td>11379.979167</td>
      <td>0.0</td>
      <td>204.645833</td>
      <td>4299.666667</td>
      <td>5744.395833</td>
      <td>287.291667</td>
      <td>841.020833</td>
      <td>17.395833</td>
      <td>224.979167</td>
      <td>1638.958333</td>
      <td>50.000000</td>
      <td>821.625000</td>
      <td>44.208333</td>
      <td>2160.833333</td>
      <td>873.666667</td>
      <td>0.0</td>
      <td>846.291667</td>
      <td>0.0</td>
      <td>119589.458333</td>
    </tr>
    <tr>
      <th>2021-02-22</th>
      <td>90202.5</td>
      <td>13329.958333</td>
      <td>0.0</td>
      <td>921.812500</td>
      <td>4294.312500</td>
      <td>5175.916667</td>
      <td>198.416667</td>
      <td>836.187500</td>
      <td>12.750000</td>
      <td>217.750000</td>
      <td>1794.166667</td>
      <td>169.125000</td>
      <td>855.458333</td>
      <td>249.750000</td>
      <td>2834.875000</td>
      <td>796.250000</td>
      <td>0.0</td>
      <td>734.875000</td>
      <td>0.0</td>
      <td>122624.104167</td>
    </tr>
    <tr>
      <th>2021-02-23</th>
      <td>90234.0</td>
      <td>7240.866667</td>
      <td>0.0</td>
      <td>329.333333</td>
      <td>4291.866667</td>
      <td>9349.200000</td>
      <td>62.000000</td>
      <td>734.266667</td>
      <td>11.466667</td>
      <td>169.800000</td>
      <td>1174.666667</td>
      <td>218.533333</td>
      <td>689.733333</td>
      <td>427.200000</td>
      <td>2899.600000</td>
      <td>622.800000</td>
      <td>0.0</td>
      <td>426.266667</td>
      <td>0.0</td>
      <td>118881.600000</td>
    </tr>
  </tbody>
</table>
<p>1882 rows × 20 columns</p>
</div>




```python
elec = make_dataset.prepare_electricity_actuals(
    r"C:\Users\rachel.hassall\git\charterful\data\raw\ELECTRICITY_ACTUALS.csv"
).drop(["INTELEC", "INTNSL", "OIL"], axis=1)

elec_as_percent_of_ted = elec.div(elec["TED"], axis=0)
elec_as_percent_of_ted = elec_as_percent_of_ted.drop("TED", axis=1)
elec_as_percent_of_ted[elec_as_percent_of_ted.index.year == 2020][
    ["CCGT", "WIND"]
].plot(figsize=(40, 20))
plt.title("Gas generation is used to counteract fluctuating wind generation")
plt.ylabel("Fraction of Total Generation per Gas Day")
plt.xlabel("Time")
plt.show()
```


    
![png](1-why-are-gas-fired-powerstations-important_files/1-why-are-gas-fired-powerstations-important_9_0.png)
    


Statistically speaking, we can prove this in a number of ways. Firstly, correlation analysis.

## Pearsons Correlation

The pearsons correlation coefficient (Pearson's r) measures linear correlation between two variables, assuming those variables are independent and normally distributed.


It's important to note that this is a measure of linear correlation, and may ignore other types of correlation. For example, below we see that the exponential relationship has quite a high Pearson's r value, despite the relationship in the data being exponential. Equally, in the piecewise figure, there are two linear piecewise relationships that the Pearson's r doesn't show.



[https://en.wikipedia.org/wiki/Pearson_correlation_coefficient]
[https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9]
[https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html]


```python
nobs = 20
x = np.linspace(0, 10, nobs)
basex = x + np.random.rand(nobs)
df = pd.DataFrame({"x": basex, "y": x, "type": ["Linear"] * nobs})


df = df.append(
    pd.DataFrame(
        {"x": basex, "y": 10 * np.power(0.5, x), "type": ["Exponential"] * nobs}
    )
)

t = np.linspace(0, 2 * np.pi, nobs)
r = [np.random.uniform(1, 5) for i in np.arange(nobs)]
df = df.append(
    pd.DataFrame(
        {"x": 5 + r * np.cos(t), "y": 5 + r * np.sin(t), "type": ["Radial"] * nobs}
    )
)

df = df.append(
    pd.DataFrame(
        {"x": basex, "y": np.where(x < 5, x, -x + 10), "type": ["Piecemeal"] * nobs}
    )
)

g = sns.lmplot(data=df, x="x", y="y", col="type", hue="type")


def annotate(data, **kws):
    r, p = scipy.stats.pearsonr(data["x"], data["y"])
    ax = plt.gca()
    ax.text(0.05, 0.8, "r={:.2f}, p={:.2g}".format(r, p), transform=ax.transAxes)


g.map_dataframe(annotate)
plt.show()
```


    
![png](1-why-are-gas-fired-powerstations-important_files/1-why-are-gas-fired-powerstations-important_12_0.png)
    


In our time series data of electricity generation from Wind and Gas, Pearson's r is a measure of global synchrony. It has a significant value of -0.5, meaning that when wind goes up, gas does down and vice versa.


```python
df = elec[["CCGT", "WIND"]].copy()

r, p = scipy.stats.pearsonr(df.dropna()["CCGT"], df.dropna()["WIND"])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
```

    Scipy computed Pearson r: -0.5361356201066011 and p-value: 1.5357043092659718e-140
    

If we normalise Gas and Wind generation by looking at it as a percentage of overall electricity generation (demand), we see the correlation increases further. This has the handy benefit of removing seasonal variations in electricity demand (high in winter, low in summer).


```python
r, p = scipy.stats.pearsonr(
    elec_as_percent_of_ted.dropna()["CCGT"], elec_as_percent_of_ted.dropna()["WIND"]
)
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
```

    Scipy computed Pearson r: -0.9616758685743998 and p-value: 7.591777434269864e-48
    

Now if we look at all components of generation as a percentage of overall electricity generation (demand) and apply the correlation calculation to give a correlation matrix with an absolute threshold of 0.4, this gives us a lot of information about the correlation between generation sources. We're primarily concerned with CCGT and OCGT generaiton, and understanding how that behaves in relation to other sources. This tells us that CCGT is inversely proportional to Wind, and Wind isn't correlated with anything else. This implies gas generation is the primary balancing measure for counteracting flexible wind generation.


```python
corr = elec_as_percent_of_ted.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr[corr.abs() > 0.4], mask=mask, vmin=-1, vmax=1)
plt.title("Electricity Generation as Percentage of Total Generation")
plt.show()
```


    
![png](1-why-are-gas-fired-powerstations-important_files/1-why-are-gas-fired-powerstations-important_18_0.png)
    


## Predictive Power Score

The problem with pearsons correlation is that it only picks up linear correlation, as demonstrated in Figure XX. Another way to test how two variables are related, is to use a Predictive Power Score (PPS). PPS detects non-linear relationships by building a model to predice one variable from another, and assessing the performance compared to a baseline.
- A score of 0 means that the column x cannot predict the column y better than a naive baseline model.
- A score of 1 means that the column x can perfectly predict the column y given the model.
- A score between 0 and 1 states the ratio of how much potential predictive power the model achieved compared to the baseline model.
[https://github.com/8080labs/ppscore]

Let's start by calculating the PPS between Wind and CCGT, the score is around 0.2 suggesting that Wind can be somewhat helpful in predicting CCGT, but by no means perfect.


```python
pps.score(elec_as_percent_of_ted, "WIND", "CCGT")
```




    {'x': 'WIND',
     'y': 'CCGT',
     'ppscore': 0.22260334348107214,
     'case': 'regression',
     'is_valid_score': True,
     'metric': 'mean absolute error',
     'baseline_score': 0.0767071613877037,
     'model_score': 0.05963189079385866,
     'model': DecisionTreeRegressor()}




```python
matr = pps.matrix(elec_as_percent_of_ted).pivot(
    columns="x", index="y", values="ppscore"
)

sns.heatmap(matr, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=False)
plt.title("Predictive Power Score")
plt.show()
```


    
![png](1-why-are-gas-fired-powerstations-important_files/1-why-are-gas-fired-powerstations-important_22_0.png)
    


## Grangers Causality

Granger's causality tests whether past values of one time series, x1, have a significantly significant effect from a sexond time series, x2.

Granger defined the causality relationship based on two principles:
- The cause happens prior to its effect.
- The cause has unique information about the future values of its effect.


We have to be careful with Granger's causality tests for a number of reasons, firstly it can be misleading when a 3rd process is driving the relationship between the variables. Secondly, the time series should be stationary, and thirdly, it is not robust to non linear relationships. We can extend the test to multivariate analysis (in our case, looking at all generation types) by using Vector Auto Regression, but for now, we'll stick with the basic causality test to start with.

[https://en.wikipedia.org/wiki/Granger_causality]


Our Null hypothesis for this test is that Wind Generation does not cause Gas Generation. This means that past values of wind do not have a statistically significant effect on the current Gas Generation value. We can reject this null hypothesis (and suggest that Wind Generation does cause Gas Generation) if the p value is below an appropriate level.

[https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html]

The Null hypothesis for grangercausalitytests is that the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1. Grange causality means that past values of x2 have a statistically significant effect on the current value of x1, taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause x1 if the pvalues are below a desired size of the test.

The null hypothesis for all four test is that the coefficients corresponding to past values of the second time series are zero.




```python
elec_as_percent_of_ted.dropna().nunique()
```




    CCGT       84
    COAL       84
    NUCLEAR    84
    WIND       84
    PS         84
    NPSHYD     84
    OCGT       84
    OTHER      84
    INTFR      84
    INTIRL     83
    INTNED     36
    INTEW      78
    BIOMASS    84
    INTNEM     84
    INTIFA2    34
    dtype: int64



The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, with the alternative that there is no unit root. If the pvalue is above a critical size, then we cannot reject that there is a unit root.

The p-values are obtained through regression surface approximation from MacKinnon 1994, but using the updated 2010 tables. If the p-value is close to significant, then the critical values should be used to judge whether to reject the null.

The autolag option and maxlag for it are described in Greene.

[https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html]


```python
res = adfuller(elec_as_percent_of_ted["CCGT"])
res
```




    (-5.055665939456727,
     1.7201055968691958e-05,
     21,
     1860,
     {'1%': -3.433870617038361,
      '5%': -2.863095154794451,
      '10%': -2.567597908717771},
     -4797.697519993104)




```python
grangercausalitytests(elec_as_percent_of_ted[["CCGT", "WIND"]].dropna(), 1)
```

    
    Granger Causality
    number of lags (no zero) 1
    ssr based F test:         F=0.5535  , p=0.4570  , df_denom=1878, df_num=1
    ssr based chi2 test:   chi2=0.5544  , p=0.4565  , df=1
    likelihood ratio test: chi2=0.5543  , p=0.4565  , df=1
    parameter F test:         F=0.5535  , p=0.4570  , df_denom=1878, df_num=1
    




    {1: ({'ssr_ftest': (0.5535386644967097, 0.456968467257192, 1878.0, 1),
       'ssr_chi2test': (0.5544229115645959, 0.4565161052169894, 1),
       'lrtest': (0.5543412198112492, 0.4565492794706799, 1),
       'params_ftest': (0.5535386644967512, 0.456968467257192, 1878.0, 1.0)},
      [<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x1f353b2df40>,
       <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x1f353b2dbe0>,
       array([[0., 1., 0.]])])}



# Outlook

Future Energy Scenarios (FES) are created to help the UK prepare for our future energy requirements, they inform investment and policy decisions from government to industry level. FES Scenarios are built using extensive data and modelling, and involve looking at consumer demand across gas, electricity, industrial, residental and transport.

[https://www.nationalgrideso.com/document/173821/download] On a high level, there are four energy scenarios.

1. Reaching net zero carbon emissions by 2050 is achievable.
However, this requires immediate action across all key
technologies and policy areas, and full engagement across
society and end consumers.
2. Hydrogen and carbon capture and storage must be deployed
for net zero. Industrial scale demonstration projects need to be
operational this decade.
3. The economics of energy supply and demand fundamentally shift
in a net zero world. Markets must evolve to provide incentives for
investment in flexibility and zero carbon generation.
4. Open data and digitalisation underpin the whole system thinking
required to achieve net zero. This is key to navigating increasing
complexity at lowest cost for consumers.

In all of these scenarios, National Grid project that Gas Fired Generation will continue to be important in some degree.

![../references/GFOP2018-GasFiredDemand-FES.png](../references/GFOP2018-GasFiredDemand-FES.png)

[https://www.elexon.co.uk/article/bsc-insight-reliant-on-the-weather-a-review-of-weather-dependent-generation-in-gb/?utm_source=Newsweaver&utm_medium=email&utm_term=BSC+Insight%3A+Reliant+on+the+weather%3F+A+review+of+weather+dependent+generation+in+GB&utm_campaign=Newscast+847+for+information]

# References

- https://www.nationalgrid.com/uk/gas-transmission/document/110176/download
- https://www.nationalgrideso.com/document/173821/download
- https://www.nationalgrideso.com/document/173796/download
- https://www.energy.gov/fe/how-gas-turbine-power-plants-work

[https://pdf.sciencedirectassets.com/277910/1-s2.0-S1876610217X00398/1-s2.0-S1876610217361155/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjENX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCTDxyepx2rxmGgOIiBmIQyGIoeE7B7%2BJp9YZ2I6oB%2B1wIhAJUWsZAryuEfovLDdBhaFmA6K%2B4c3PYfpI2ecBS7lIBRKrQDCC4QAxoMMDU5MDAzNTQ2ODY1Igwkgdbmv4trglf7uzoqkQO4LUPtCHIyB%2FEAJ3g6Nv4l%2FBIwmwEuGeL%2BGNoXkTxNHFB7EljBebOmbEFd5OFvU4qkHnkbUJvxYihCU0MTpn8wIBTi2MkbfX6aLDVlAn37kqmeqkQorgwdyhROIjbfsMAjAIjInni5KErtamj9TUiLwRuqY92c0ddsDtUSsVrgD3ok2bCkET5O8WAdcqLKCWZ5qnuAhNRQoaA3hqT8tIyQBzbGRpxUeVqZqCj0xXJgvuA2Xf1wYDoqO3zVTExwyC0scMSMjFvHFCdz6tyKRvaSkYVakYXA6rq0iNWzO0Rcz2HcuQqyBVUI8VYZNnruFjqRggq%2FaZdK6mnl6x0iM0qZgg%2BMkj%2FzB9L7IN3mR8qpXRmpvMdVmlF3aPH%2FMdxkEakgoHFSBJoXW0XAZTUbjuYkFXSnAMKKACPTPdn9glgj5u8LXCO7I6%2FRFnpxckT5rde30c%2FuBB8yMxxCGqCKsWleA%2BF9NpaxCuhu2Mi5mKabnfeIcGtmJANjqlRAw3yiUqt%2FX4m0uwn6myzhOZmrstcNKTCHrbGDBjrqAbH2YgY5%2FnRqqXx6Vjl01w90shLm1GChB9lWJjcC6tJVqSRHuXmfL3aFmADXaFEp0EbWlE5QReplA7Yjq9AeSmG6II0ld87PFCEgyQJeb2qqL6LNypJQ%2FNRZGTPFDH0kbC2%2FdvHqGjgX5VWrEmn6yqSuzHRf%2F%2Fe3Hj5k80Twmnp%2BNlXEr0B6QtRN8xLL5gTBvQzme7A3o6x%2BvZ0HSPWuVRMfPCc9y4h3jF24yu3EM80kcA3noTyeAEYVBrfw4VwsWUYtEKl8kvimet6YiKySXwvFSLRcugjf9sbsZX03j2TNgTIwBjLkvbRwAA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210406T134157Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4NJQE54Q%2F20210406%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a7a2c1d14aeb76c8a894d5a5d5118775df6193a3d76a38c2fdc734ef57d2bc22&hash=a47bbfee9200c399a31635e6fdb0aa55e484374b8a8b1184738b531b7bd6d924&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1876610217361155&tid=spdf-e88c249e-1080-44a3-b7b4-339ed1d2f6c0&sid=e9a76cbf5c2d4948955b5108b0d45af45e52gxrqa&type=client]


```python

```
