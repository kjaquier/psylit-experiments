"""
This module provide tools to work with information cascades as time-stamped binary sequences.
It is designed to be generic as much as possible, but some things were hard-coded
when convenient. It has to do mostly with index levels.
"""
import itertools as it
import pathlib
import sys
import logging
from functools import partial

from IPython.display import display
import pandas as pd
import numpy as np
import dask

from utils.misc import progress
from utils.io import file_parts

ROOT = pathlib.Path()
DATA_ROOT = ROOT  / 'data' / 'processed' / 'train'

ROLE_LEVEL, ENT_LEVEL, FEAT_LEVEL = 0, 1, 2

logger = logging.getLogger()


class Cascades:
    """Wraps a pandas.DataFrame with index levels [Subject, t] and binary columns."""
    
    def __init__(self, casc, copy=True):
        if isinstance(casc, pd.DataFrame):
            self.casc = casc.copy() if copy else casc
            self.stashed = {}
        else:
            self.casc = casc.casc.copy() if copy else casc
            self.stashed = {k: v.copy() for k, v in casc.stashed.items()} if copy else dict(casc.stashed)
        
    @staticmethod
    def from_raw_df(df, document_name=None, document_col='Document'):
        df = df.copy()
        for c in [c for c in df.columns if c.endswith('_nan')]:
            df.pop(c)
        if document_name:
            df[document_col] = document_name
            df.set_index(document_col, append=True, inplace=True)
        casc = Cascades(df)
        casc.fill_gaps()
        return casc
        
    @staticmethod
    def from_csv(csv_path, add_document_index=False, **kwargs):
        document_name = file_parts(csv_path)[0] if add_document_index else None

        def get_dtype(col_name):
            fixed = {'Subject': str, 't': int, 'Document': str, 'book': str}
            dtype = fixed.get(col_name, np.int8)
            return dtype

        col_names = pd.read_csv(csv_path, nrows=0, engine='python').columns
        
        dtype_map = {k: get_dtype(k) for k in col_names}
        kwargs = {
            **dict(
                dtype=dtype_map,
                index_col=['Subject', 't'],
            ),
            **kwargs
        }
        raw_df = pd.read_csv(csv_path, **kwargs)
        return Cascades.from_raw_df(raw_df, document_name=document_name)
        
    def to_csv(self, *args, **kwargs):
        return self.casc.to_csv(*args, **kwargs)
    
    def _repr_html_(self):
        return display(self.casc)
    
    def stash(self, cols):
        for c in cols:
            self.stashed[c] = self.casc.pop(c)

    def retrieve(self, cols):
        for c in cols:
            self.casc[c] = self.stashed.pop(c)

    def remove_cols(self, cols):
        for c in cols:
            if c in cols:
                self.casc.pop(c)

    def match_cols(self, *match_funcs):
        return [c for c in self.casc.columns if all(f(c) for f in match_funcs)]
        
    def split_cols(self, split='_', default=''):

        def col_len(col):
            if isinstance(col, tuple):
                return len(col)
            else:
                return 1

        def split_col(col):
            if isinstance(col, tuple):

                for lvl in col:
                    yield from lvl.split(split)
            else:
                col_tuple = col.split(split)
                if len(col_tuple) == 1:
                    yield col
                else:
                    yield from col_tuple

        def pad_col(col, n):
            col = tuple(col)
            m = len(col)
            return tuple(col + (default,) * (n-m))

        n = max(col_len(c) for c in self.casc.columns) + 1
        col_tuples = [pad_col(split_col(c), n) for c in self.casc.columns]
        self.casc.columns = pd.MultiIndex.from_tuples(col_tuples)
    
    def group_rows(self, columns, sort_by=None, agg_func='sum'):
        c = self.casc.groupby(columns).agg(agg_func)
        if sort_by:
            return c.sort_index(level=sort_by)
        return c

    #def remap_columns(self, mapper, merge='any'):
    #    self.casc.groupby(mapper, )
    #    self.casc.columns = pd.MultiIndex.from_tuples(map(mapper, self.casc.columns))


    def group_columns(self, by, func, use_dask=False):
        df = dask.dataframe(self.casc) if use_dask else self.casc
        processed = (df
                     .groupby(by, axis='columns')
                     .apply(lambda c: c.agg(func, axis=1).astype(np.int8)))
        if use_dask:
            processed = processed.compute()
        new_casc = Cascades(processed)

        return new_casc

    def set_column_levels(self, index_level_names):
        self.casc.columns.set_names(index_level_names, inplace=True)

    def rename_cols(self, columns_mapper):
        '''columns_mapper: dict-like or function'''
        self.casc.rename(index=str, columns=columns_mapper, inplace=True)
    
    def fill_gaps(self):
        # Note: doesn't work with books
        t_all = self.casc.reset_index(['Subject', 't']).t
        t_unique = t_all.unique()
        
        new_index_t = pd.Index(t_unique, name="t")

        casc = (self.casc
                .reset_index('t')
                .groupby(level='Subject', group_keys=True)
                .apply(lambda c: c.set_index('t')
                                  .reindex(new_index_t, fill_value=0)))
        return Cascades(casc)

    def fill_all(self):
        # Note: doesn't work with books
        t_all = self.casc.reset_index(['Subject', 't']).t
        t_unique = t_all.unique()
        
        t_max = t_unique.max()
        dt_min = np.diff(t_unique).min() # assume sorted
        time_range = np.arange(0, t_max+1, dt_min)
        new_index_t = pd.Index(time_range, name="t")
        
        casc = (self.casc
                .reset_index('t')
                .groupby(level='Subject', group_keys=True)
                .apply(lambda c: c.set_index('t')
                                  .reindex(new_index_t, fill_value=0)))
        return Cascades(casc)
    
    def pair(self, sources, destinations, keep_single=(), split=' & ', name_fmt="{src}{split}{dst}"):
        casc = self.casc
        pairs = [
            (s, d)
            for s in sources
            for d in destinations
            if s not in keep_single
            if d not in keep_single
        ]
        df = pd.DataFrame({
            #name_fmt.format(src=s, split=split, dst=d): casc[s] * casc[d]
            (s, d): casc[s] * casc[d]
            for s, d in pairs})
        for c_name in keep_single:
            df[c_name] = casc[c_name]
        
        df.columns = pd.MultiIndex.from_tuples(list(df.columns))
        return Cascades(df)

    def pair_dask(self, sources, destinations, keep_single=(), split=' & ', name_fmt="{src}{split}{dst}"):
        casc = self.casc
        pairs = [
            (s, d)
            for s in sources
            for d in destinations
            if s not in keep_single
            if d not in keep_single
        ]
        pair_cascs = lambda c, s, d: {(s, d): c[s] * c[d]}
        paired = [dask.delayed(pair_cascs)(casc, s, d) for s, d in pairs]
        df = dask.dataframe.from_delayed(paired)
        df.assign(**{
            c_name: casc[c_name]
            for c_name in keep_single
        })

        df.columns = pd.MultiIndex.from_tuples(list(df.columns))
        return df

    def memory_size(self):
        return sys.getsizeof(self.casc)
    
    @property
    def n_rows(self):
        return len(self.casc.index)
    
    @property
    def subjects(self):
        return self.casc.index.get_level_values('Subject').unique().array
    
    @staticmethod
    def concat(cascades, document_values, document_col):
        cs = []
        for cascade, discr in zip(cascades, document_values):
            c = cascade.casc.assign(**{document_col: discr})
            c = c.set_index(document_col, append=True)
            cs.append(c)
        return pd.concat(cs, sort=False)

    def batch_single_measure(self, trajectory_group, measure, measure_name, k_values=(1,), local=False, window_size=1):
        casc = self.casc
        B = []
        trajectory_group = [trajectory_group] if isinstance(trajectory_group, str) else list(trajectory_group)
        #cols = [*trajectory_group, *casc.columns.names, 'k', measure_name]
        for lbl, df in progress(casc.groupby(level=trajectory_group), print_func=logger.debug):
            lbl = [lbl] if isinstance(lbl, str) else list(lbl)
            if window_size > 1:
                df = ((df.rolling(window=window_size).sum() > 0)
                      .astype(np.int8)
                      .fillna(0))
            for c in progress(df.columns, print_func=logger.debug):
                for k in k_values:
                    kwargs = dict(k=k, local=local)
                    meas = measure(df[c], **kwargs)
                    B.append({**{k: v for k, v in zip(trajectory_group, lbl)},
                              **{k: v for k, v in zip(casc.columns.names, c)},
                              **meas})
        return pd.DataFrame(B)#, columns=cols)

    def batch_pairwise_measure(self, trajectory_group, measure, measure_name,
                               src_cols=None, dest_cols=None, window_size=1,
                               get_args=(lambda src, dst: {})):
        casc = self.casc
        src_cols = src_cols or casc.columns.values
        dest_cols = dest_cols or casc.columns.values

        many_results = not isinstance(measure_name, str)

        col_pairs = list(it.product(src_cols, dest_cols))
        logger.debug("%d x %d = %d column pairs", len(src_cols), len(dest_cols), len(col_pairs))

        trajectory_group = [trajectory_group] if isinstance(trajectory_group, str) else list(trajectory_group)
        rows = []
        for lbl, df in progress(casc.groupby(level=trajectory_group), print_func=logger.debug):
            lbl = [lbl] if isinstance(lbl, str) else list(lbl)
            if window_size > 1:
                df = ((df.rolling(window=window_size).sum() > 0)
                      .astype(np.int8)
                      .fillna(0))

            for src, dst in col_pairs:
                kwargs = get_args(src, dst)
                meas = measure(df[src], df[dst], **kwargs)
                if many_results:
                    row = {'Source': src, 'Target': dst, 
                           **dict(zip(measure_name, meas))}
                else:
                    row = {'Source': src, 'Target': dst, measure_name: meas}
                rows.append(row)
        return pd.DataFrame(rows).set_index(['Source', 'Target'])

    def batch_pairwise_df_measure(self, trajectory_group, measure,
                                  src_cols=None, dest_cols=None, window_size=1,
                                  get_args=(lambda src, dst: {})):
        casc = self.casc
        src_cols = list(src_cols or casc.columns.values)
        dest_cols = list(dest_cols or casc.columns.values)

        col_pairs = list(it.product(src_cols, dest_cols))
        logger.debug("%d x %d = %d column pairs", len(src_cols), len(dest_cols), len(col_pairs))

        trajectory_group = [trajectory_group] if isinstance(trajectory_group, str) else list(trajectory_group)
        rows = []
        for lbl, df in casc.groupby(level=trajectory_group):#progress(, print_func=logger.debug):
            lbl = [lbl] if isinstance(lbl, str) else list(lbl)
            if window_size > 1:
                df = ((df.rolling(window=window_size).sum() > 0)
                      .astype(np.int8)
                      .fillna(0))

            for src, dst in col_pairs:
                kwargs = get_args(src, dst)
                meas = measure(df, src, dst, **kwargs)
                rows.append({'Source': src, 'Target': dst, **meas})
        return pd.DataFrame(rows).set_index(['Source', 'Target'])

    def batch_multi_df_measure(self, trajectory_group, measure,
                               cols=None, window_size=1,
                               get_args=(lambda src, dst: {})):
        casc = self.casc
        cols = list(cols or casc.columns.values)

        logger.debug("%d columns", len(cols))

        trajectory_group = [trajectory_group] if isinstance(trajectory_group, str) else list(trajectory_group)
        rows = []
        for lbl, df in progress(casc.groupby(level=trajectory_group), print_func=logger.debug):
            lbl = [lbl] if isinstance(lbl, str) else list(lbl)
            if window_size > 1:
                df = ((df.rolling(window=window_size).sum() > 0)
                      .astype(np.int8)
                      .fillna(0))

            kwargs = get_args(cols)
            meas_rows = measure(df, cols, **kwargs)
            rows.append({'Subject': lbl, **r} for r in meas_rows)

        return pd.DataFrame(rows).set_index(['Source', 'Target'])


class MultiCascades:
    """Cascades from multiple documents, distinguished by an additional index level"""

    def __init__(self, cascades):
        self.casc = cascades
        
    def get_dt(self, group_by=('Subject', 'Document')): # if document present should group by document
        return (self.casc
                .reset_index('t')[['t']]
                .groupby(level=list(group_by))
                .transform(pd.Series.diff)
                .t)
    
    @classmethod
    def from_cascades(cls, cascades, document_values, document_col='Document'):
        return cls(Cascades.concat(cascades, document_values, document_col))

    @classmethod
    def from_csvs(cls, files, document_col='Document'):
        doc_name = lambda f: file_parts(f)[0]
        return cls.from_cascades(
            (Cascades.from_csv(f) for f in progress(files)),
            document_values=list(map(doc_name, files)),
            document_col=document_col,
        )

    def pair_dask(self, sources, destinations, keep_single=(), split=' & ', name_fmt="{src}{split}{dst}"):
        casc = Cascades(self.casc)
        df = casc.pair_dask(sources, destinations, keep_single, split, name_fmt)
        return DaskMultiCascades(df)

    def pair(self, sources, destinations, keep_single=(), split=' & ', name_fmt="{src}{split}{dst}"):
        # Copy of Cascades.pair
        casc = self.casc.copy()
        singles = [(c, casc.pop(c)) for c in keep_single]
        pairs = list(it.product(sources, destinations))
        df = pd.DataFrame({
            #name_fmt.format(src=s, split=split, dst=d): casc[s] * casc[d]
            (s, d): casc[s] * casc[d]
            for s, d in pairs})
        for c_name, c in singles:
            df[c_name] = c
        
        df.columns = pd.MultiIndex.from_tuples(list(df.columns))
        return Cascades(df)

    def _repr_html_(self):
        return display(self.casc)
    
    def head(self, n):
        return self.casc.head(n)

    def memory_size(self):
        return sys.getsizeof(self.casc)
    
    @property
    def n_rows(self):
        return len(self.casc.index)
    
    @property
    def subjects(self):
        return self.casc.index.get_level_values('Subject').unique().array
    
    def match_cols(self, *match_funcs):
        # copy of Cascades.match_cols
        return [c for c in self.casc.columns if all(f(c) for f in match_funcs)]


class DaskMultiCascades:

    def __init__(self, df):
        self.df = df

    @classmethod
    def from_csvs(cls, files, document_col='Document'):
        return DaskMultiCascades(dd.read_csv(files, include_path_column=document_col, compression='zip', blocksize=None))

    @property
    def n_rows(self):
        return self.df.groupby(['Document', 'Subject']).size().compute()

    @property
    def subjects(self):
        return self.df[['Document', 'Subject']].drop_duplicates()
            

def map_col_to_stimulus_response(col):
    role, ent, feat = col
    if not ent and not feat:
        return ('Other', role)
    if (role, ent) == ('Agent', 'Subject'):
        return ('Response', feat)
    return ('Stimulus', feat)


def features_transform(casc, column_grouper, use_dask=False):
    keep_single = ['neg', 'R_unknown']
    #name_fmt=lambda src, dst: f"{src[2:]} {dst[2:]}".replace('_',' ').title())
    casc = casc.pair(casc.match_cols(lambda c: c.startswith('R_') and c not in keep_single),
                     casc.match_cols(lambda c: c.startswith(
                         'L_') and c not in keep_single),
                     keep_single=keep_single)
    # casc = Cascades(casc.compute())
    casc.rename_cols(lambda c: c.replace('R_', '').replace('L_', '').title())
    casc.split_cols()

    def str_join_grouper_decorator(col_tuple):
        return '_'.join(column_grouper(col_tuple))

    casc = casc.group_columns(str_join_grouper_decorator, 'any', use_dask=use_dask)
    casc.split_cols()
    casc.set_column_levels(['Category', 'Feature'])
    casc.remove_cols([
        ('Response', 'Positive'),
        ('Response', 'Negative'),
        ('Stimulus', 'Positive'),
        ('Stimulus', 'Negative'),
        ('Other', 'Neg'),
    ])
    return casc


def transform_to_stimulus_response(casc, use_dask=False):
    return features_transform(casc, map_col_to_stimulus_response, use_dask=use_dask)

