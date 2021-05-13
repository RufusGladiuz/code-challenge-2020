from dataclasses import dataclass

@dataclass
class ProcessSettings:
    """Simple data class to hold setting the data processing

        Parameters
        ----------
            label_column: str -> Name of the columne that hold the prediction values of the dataset.
            drop_duplicates: list -> List of column names to consider for dropping duplicates
            test_size: float = 0.2 -> Size of the test dataset after the test train split.
            drop: list = [] -> List of column names that should be excluded from the final dataframe.
            to_one_hot_encode: list = [] -> List of column names that should get an one-hot-encoded.
            to_ordinal_encode: list = [] -> List of column names that should get an ordinal encoding.
            fill_mean: list = [] -> List of column names of which the nan values should get filled with the mean of that column.
            normalize: bool = Flase -> To normalize the dataset or not.
            fill_nan: float =  -1 -> Value to full Nan values with.
            text_column: str -> Name of the column that hold text and should get converted to a ML suitable format.
            nlp_tool: str -> Input need to be either 'tf-idf' (Text frequency, inverse document frequency) or 'bow' (Bag of Words). Will process the column chosen in text_column.
            tf_idf_cutoff: float -> tf-idf values below the input values well get neglected.
            tokenizer -> Optinoal Tokenizer object for tokenizing texts.
            stop_words: list -> List of stopwords to removed from the text. Standard is nltk english stopwords.
            token_pattern: str -> A regular expression to identify word tokens. Will be ignored if tokenizer is not None.
            save_dir: str -> A directory to save the dataset to.
    """
    label_column: str
    drop_duplicates: list = []
    test_size = 0.2,
    drop: list = [],
    to_one_hot_encode: list = [],
    to_ordinal_encode: list = [],
    fill_mean: list = [],
    normalize = False,
    fill_nan: float = -1,
    text_column: str = "",
    nlp_tool: str = None,
    tf_idf_cutoff: float = 0,
    tokenizer = None,
    stop_words: list = None,
    token_pattern: str = "[A-Za-z]+",
    save_dir: str = None