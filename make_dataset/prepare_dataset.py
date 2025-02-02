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

from process_settings import ProcessSettings

def join_one_hot_encoding(data: pd.DataFrame, column: str) -> list:
    """ Creates one-hot-encodings for a column and drops that column.
    
        Parameters
        ----------
            data: pd.dataFrame -> DataFrame of the main data source
            column: str -> Name of the column to one-hot-encode
        
        Returns
        -------
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
    
        Parameters
        ----------
            X_train: pd.DataFrame -> Training dataset
            X_test: pd.DataFrame -> Test dataset
            to_encode: list -> List of columns to encode
    
        Returns
        -------
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
    
        Parameters
        ----------
            data: pd.DataFrame -> Pandas Dataframe containing a text column.
            column_name: str -> Name of column holding text.
            vectorizer -> Object of Sk-Learn libary to convert text features.
    
        Returns
        -------
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
        
        Parameters
        ----------
            X_train: pd.DataFrame -> Train dataset as a pd.DataFrame.
            X_test: pd.DataFrame -> Test dataset as a pd.DataFrame.
            exclude_columns: list -> Name of columns to exclude from the normalization.
    
        Returns
        -------
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
                 settings: ProcessSettings,
                 quantile_nan_fill: list = []) -> pd.DataFrame:
    """Creates the Wine-Score dataset. However can be used to process simular datasets.
    
        Parameters
        ----------
            load_dir: str -> Directory to the raw data container as a csv.
            settings: ProcessSettings -> Settings to use for the processing of the data
            quantile_nan_fill: list = [] -> List with objects of class QuantileCutOrder to determine of which columns the nan values should be filled by a quantile of a categorical value.
        Returns
        -------
            Four dataframes X_train, X_test, y_train, y_test.
    """

    # Load Dataset, drop duplicates and drop all columns with no label.
    data = pd.read_csv(load_dir)
    data = data.drop_duplicates(settings.drop_duplicates)
    data = data.dropna(subset=[settings.label_column])

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
    for to_fill in settings.fill_mean:
        data[to_fill] = data[to_fill].fillna(data[to_fill].mean())

    data = data.drop(columns=settings.drop, errors="ignore")
    data = data.reset_index(drop=True)

    for column in settings.to_one_hot_encode:
        data = join_one_hot_encoding(data, column)

    # Split dataset into test and train data
    X_train, X_test, y_train, y_test = train_test_split(data.drop(
        columns=[settings.label_column]), data[settings.label_column], test_size=settings.test_size, random_state=0)
    
    if len(settings.to_ordinal_encode) > 0:
        X_train, X_test = oridnal_encoding(X_train, X_test, settings.to_ordinal_encode)

    # Converting text features
    if settings.nlp_tool is None:
        X_train = X_train.drop(columns=[settings.text_column])
        X_test = X_test.drop(columns=[settings.text_column])
    else:
        X_train[settings.text_column] = X_train[settings.text_column].apply(
            lambda txt: txt.lower())
        X_test[settings.text_column] = X_test[settings.text_column].apply(
            lambda txt: txt.lower())
        corpus = X_train[settings.text_column]

        if settings.nlp_tool == "tf-idf":
            vectorizer = TfidfVectorizer(tokenizer=settings.tokenizer,
                                         use_idf=True,
                                         stop_words=settings.stop_words,
                                         min_df=settings.tf_idf_cutoff,
                                         token_pattern=settings.token_pattern)
        if settings.nlp_tool == "bow":
            vectorizer = CountVectorizer(
                tokenizer=settings.tokenizer,
                stop_words=settings.stop_words,
                token_pattern=settings.token_pattern)

        vectorizer = vectorizer.fit(corpus)

        X_train = _apply_nlp_transformation(X_train, settings.text_column, vectorizer)
        X_test = _apply_nlp_transformation(X_test, settings.text_column, vectorizer)

        X_train = X_train.fillna(settings.fill_nan)
        X_test = X_test.fillna(settings.fill_nan)

    if settings.normalize:
        X_train, X_test = normalize_features(
            X_train, X_test, settings.to_one_hot_encode)

    if settings.save_dir is not None:

        if settings.save_dir[-1] == "/" or settings.save_dir[-1] == "\\":
            save_dir = settings.save_dir[len(settings.save_dir) - 1]

        X_train.join(y_train).to_csv(
            f"{save_dir}/train_set_{settings.label_column}.csv")
        X_test.join(y_test).to_csv(f"{save_dir}/test_set_{settings.label_column}.csv")

    return X_train, X_test, y_train, y_test
