import pandas as pd

from numpy import where as numpy_where

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import(
    TfidfVectorizer,
    CountVectorizer
)

from sklearn.preprocessing import(
    OrdinalEncoder,
    StandardScaler
)

from nltk.corpus import stopwords


def join_one_hot_encoding(data: pd.DataFrame, column: str) -> list:
    """ Creates one-hot-encodings for a column and drops that column.
    
        Args:
            data: pd.dataFrame -> DataFrame of the main data source
            column: str -> Name of the column to one-hot-encode
        
        Returns:
            Returns the dataframe with one-hot-encodings
    
    """
    data = data.join(pd.get_dummies(data[column], prefix=column))
    data = data.drop(columns=[column])
    return data

def oridnal_encoding(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        to_encode: list):
    """Ordinaly encodes chosen columns
    
        Args:
            X_train: pd.DataFrame -> Training dataset
            X_test: pd.DataFrame -> Test dataset
            to_encode: list -> List of columns to encode
    
        Returns
            Train and Test set with ordinal encodings
    """
    enc = OrdinalEncoder(unknown_value=-1, handle_unknown="use_encoded_value")
    enc.fit([a for a in X_train[to_encode].fillna("filler_").values])

    trans_train = enc.transform(
        [a for a in X_train[to_encode].fillna("filler_").values])

    trans_test = enc.transform(
        [a for a in X_test[to_encode].fillna("filler_").values])

    filler_values = []

    # Finds and replaces all Nan values with -1
    for cat_collection in enc.categories_:
        if len(numpy_where(cat_collection == "filler_")[0]) != 0:
            filler_values.append(
                int(numpy_where(cat_collection == "filler_")[0]))
        else:
            filler_values.append(-100)

    trans_train = _replace_nan_values_ordinal_encoding(
        trans_train, filler_values)
    trans_test = _replace_nan_values_ordinal_encoding(trans_test, filler_values)
    X_train[to_encode] = trans_train
    X_test[to_encode] = trans_test

    return X_train, X_test


def _replace_nan_values_ordinal_encoding(values: list, filler_numbers) -> list:
    for vals in values:
        for i in range(len(vals)):
            if filler_numbers[i] == vals[i]:
                vals[i] = -1

    return values



def _apply_nlp_transformation(
        data: pd.DataFrame,
        column_name: str,
        vectorizer) -> pd.DataFrame:
    """ Applies text feature conversion to a column.
    
        Args:
            data: pd.DataFrame -> Pandas Dataframe containing a text column.
            column_name: str -> Name of column holding text.
            vectorizer -> Object of Sk-Learn libary to convert text features.
    
        Returns:
            A dataframe with converted text features
    """
    vectorized_text = vectorizer.transform(data[column_name])
    data = data.drop(columns=[column_name])
    data = data.join(pd.DataFrame(vectorized_text.toarray()))
    return data


def normalize_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        exclude_columns: list) -> pd.DataFrame:
    """Normalizes features in fashion so that one-hot-encoded features are not affected.
        
        Args:
            X_train: pd.DataFrame -> Train dataset as a pd.DataFrame.
            X_test: pd.DataFrame -> Test dataset as a pd.DataFrame.
            exclude_columns: list -> Name of columns to exclude from the normalization.
    
        Returns:
            Scaled Train and Test Datasets.
    
    """
    columns = list(X_train.columns)
    no_scale = []

    # Find all one-hot-encoded columns
    for one_hot in exclude_columns:
        no_scale_temp = [col for col in columns if one_hot in str(col)]
        no_scale += no_scale_temp
    
    # Find all columns to encode
    to_scale = [col for col in columns if col not in no_scale]
    scaler = StandardScaler()
    scaler.fit(X_train[to_scale])
    X_train[to_scale] = scaler.transform(X_train[to_scale])
    X_test[to_scale] = scaler.transform(X_test[to_scale])

    return X_train, X_test


def process_data(load_dir: str,
                 label_column: str,
                 drop_duplicates: list,
                 test_size = 0.2,
                 drop: list = [],
                 quantile_nan_fill: list = [],
                 to_one_hot_encode: list = [],
                 to_ordinal_encode: list = [],
                 fill_mean: list = [],
                 normalize = False,
                 fill_nan: float = -1,
                 text_column: str = "",
                 nlp_tool: str = None,
                 tf_idf_cutoff: float = 0,
                 tokenizer = None,
                 stop_words: list = stopwords.words('english'),
                 token_pattern: str = "[A-Za-z]+",
                 save_dir: str = None) -> pd.DataFrame:
    """Creates the Wine-Score dataset. However can be used to process simular datasets.
    
        Args:
            load_dir: str -> Directory to the raw data container as a csv.
            label_column: str -> Name of the columne that hold the prediction values of the dataset.
            drop_duplicates: list -> List of column names to consider for dropping duplicates
            test_size: float = 0.2 -> Size of the test dataset after the test train split.
            drop: list = [] -> List of column names that should be excluded from the final dataframe.
            quantile_nan_fill: list = [] -> List with objects of class QuantileCutOrder to determine of which columns the nan values should be filled by a quantile of a categorical value.
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
    
        Returns:
            Four dataframes X_train, X_test, y_train, y_test.
    """

    # Load Dataset, drop duplicates and drop all columns with no label.
    data = pd.read_csv(load_dir)
    data = data.drop_duplicates(drop_duplicates)
    data = data.dropna(subset=[label_column])

    # If quantile_nan_fill is set. The chosen column gets filled with the average values considering a categorical feature.

    for order in quantile_nan_fill:
        for cat in data[order.per_category].unique():
            quantile = data[data[order.per_category] ==
                            cat][order.outlier_column].quantile(order.quantile)

            cat_mean = data[(data[order.per_category] == cat) & (
                data[order.outlier_column] < quantile)][order.outlier_column].mean()

            data.loc[data[order.per_category] == cat, order.outlier_column] = data[data[order.per_category]
                                                                                   == cat][order.outlier_column].fillna(cat_mean)

    # Filling nan values of chosen columns with there mean.
    for to_fill in fill_mean:
        data[to_fill] = data[to_fill].fillna(data[to_fill].mean())

    data = data.drop(columns=drop, errors="ignore")
    data = data.reset_index(drop=True)

    for column in to_one_hot_encode:
        data = join_one_hot_encoding(data, column)

    # Split dataset into test and train data
    X_train, X_test, y_train, y_test = train_test_split(data.drop(
        columns=[label_column]), data[label_column], test_size=test_size, random_state=0)
    
    if len(to_ordinal_encode) > 0:
        X_train, X_test = oridnal_encoding(X_train, X_test, to_ordinal_encode)

    # Converting text features
    if nlp_tool is None:
        X_train = X_train.drop(columns=[text_column])
        X_test = X_test.drop(columns=[text_column])
    else:
        X_train[text_column] = X_train[text_column].apply(
            lambda txt: txt.lower())
        X_test[text_column] = X_test[text_column].apply(
            lambda txt: txt.lower())
        corpus = X_train[text_column]

        if nlp_tool == "tf-idf":
            vectorizer = TfidfVectorizer(tokenizer=tokenizer,
                                         use_idf=True,
                                         stop_words=stop_words,
                                         min_df=tf_idf_cutoff,
                                         token_pattern=token_pattern)
        if nlp_tool == "bow":
            vectorizer = CountVectorizer(
                tokenizer=tokenizer,
                stop_words=stop_words,
                token_pattern=token_pattern)

        vectorizer = vectorizer.fit(corpus)

        X_train = _apply_nlp_transformation(X_train, text_column, vectorizer)
        X_test = _apply_nlp_transformation(X_test, text_column, vectorizer)

        X_train = X_train.fillna(fill_nan)
        X_test = X_test.fillna(fill_nan)

    if normalize:
        X_train, X_test = normalize_features(
            X_train, X_test, to_one_hot_encode)

    if save_dir is not None:

        if save_dir[-1] == "/" or save_dir[-1] == "\\":
            save_dir = save_dir[len(save_dir) - 1]

        X_train.join(y_train).to_csv(
            f"{save_dir}/train_set_{label_column}.csv")
        X_test.join(y_test).to_csv(f"{save_dir}/test_set_{label_column}.csv")

    return X_train, X_test, y_train, y_test
