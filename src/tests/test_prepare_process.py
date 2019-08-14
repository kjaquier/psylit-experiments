# pylint: disable=redefined-outer-name

import pathlib

import pytest
import pandas as pd

from src.data.preprocess import read_pg
from src.models import nlp_pipeline
from src.features.gen_cascades import BookData


DATA_ROOT = pathlib.Path('testdata')
TEST_BOOK_NAME = 'testbook'


@pytest.fixture
def data_text():
    return read_pg(DATA_ROOT /'testbook.txt')


@pytest.fixture
def data_interim():
    return pd.read_csv(DATA_ROOT / 'testbook_interim.csv', index_col=0)


@pytest.fixture
def data_processed():
    return pd.read_csv(DATA_ROOT / 'testbook_processed.csv', index_col=0)


@pytest.fixture
def nlp():
    nlp_pipeline.make_nlp()


def test_data_prepare(nlp, data_text, data_interim):
    pipeline = nlp_pipeline.BookParsePipeline(nlp,
                                              save_entities=False,
                                              save_data=False,
                                              save_doc=False,
                                              save_features=False)

    pipeline.parse_batches(data_text)

    data_df = pipeline.data['data_df']
    pd.testing.assert_frame_equal(data_df, data_interim)

@pytest.mark.parametrize('min_occ', [2])
def test_data_process(data_processed, min_occ):
    book = BookData(TEST_BOOK_NAME, DATA_ROOT)
    cascades = book.get_all_cascades(min_entities_occurrences=min_occ)

    assert (cascades.groupby('Subject').count() > min_occ).all()

    pd.testing.assert_frame_equal(cascades, data_processed)
