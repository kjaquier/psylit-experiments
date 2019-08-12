import pathlib

import pytest
import pandas as pd

from src.data.preprocess import read_pg
from src.models import nlp_pipeline
from src.features.gen_cascades import BookData

DATA_ROOT = pathlib.PurePath('testdata')
TEST_BOOK_NAME = 'testbook'


@pytest.fixture
def data_text():
    return read_pg(DATA_ROOT /'testbook.txt')


@pytest.fixture
def data_interim():
    return pd.read_csv(DATA_ROOT / 'testbook_interim.csv')


@pytest.fixture
def data_processed():
    return pd.read_csv(DATA_ROOT / 'testbook_processed.csv')


@pytest.fixture
def nlp():
    nlp_pipeline.make_nlp()


def test_data_prepare(nlp, data_text, tmp_path, data_interim):
    pipeline = nlp_pipeline.BookParsePipeline(nlp,
                                              output_dir=tmp_path,
                                              run_name=TEST_BOOK_NAME,
                                              save_entities=False,
                                              save_data=True,
                                              save_doc=False,
                                              save_features=False)

    pipeline.parse_batches(data_text)

    data_df = pd.read_csv(pipeline.get_output_prefix() + '.data.csv')
    pd.testing.assert_frame_equal(data_df, data_interim)

@pytest.mark.parametrize('min_occ', [2])
def test_data_process(data_text, tmp_path, data_processed, min_occ):
    book = BookData(TEST_BOOK_NAME, DATA_ROOT)
    cascades = book.get_all_cascades(min_entities_occurrences=min_occ)

    assert (cascade.groupby('Subject').count() > min_occ).all()

    pd.testing.assert_frame_equal(cascade, data_processed)
