import pandas as pd
import numpy as np


xs = pd.Series((np.random.random([1000]) > .3).astype(np.int32))
fake = pd.DataFrame({
    'X': xs,
    'Y': xs.shift(periods=-1, fill_value=0),
    'Z': xs.shift(periods=-2, fill_value=0),
})
print(fake.head(20))

from models.info_dynamics import apparent_tfr_entropy
res = apparent_tfr_entropy(fake, 'X', 'Y', k=10)
print('X->Y', res)
res = apparent_tfr_entropy(fake, 'Y', 'X', k=10)
print('Y->X', res)