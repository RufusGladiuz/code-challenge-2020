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

#TODO: Add Comments

def join_one_hot_encoding(df:pd.DataFrame, column:str) -> list:
        df = df.join(pd.get_dummies(df[column], prefix=column))
        df = df.drop(columns = [column])
        return df

def replace_nan_values_ordinal_encoding(values:list, filler_numbers) -> list:
    for vals in values: 
        for i in range(len(vals)):
            if filler_numbers[i] == vals[i]:
                vals[i] = -1

    return values

def oridnal_encoding(X_train:pd.DataFrame, X_test:pd.DataFrame, to_encode:list):
    enc = OrdinalEncoder(unknown_value = -1, handle_unknown = "use_encoded_value")
    enc.fit([a for a in X_train[to_encode].fillna("filler_").values])
    trans_train = enc.transform([a for a in X_train[to_encode].fillna("filler_").values])
    trans_test = enc.transform([a for a in X_test[to_encode].fillna("filler_").values])

    filler_values = []

    for cat_collection in enc.categories_:
        if len(numpy_where(cat_collection=="filler_")[0]) is not 0:
            filler_values.append(int(numpy_where(cat_collection=="filler_")[0]))
        else:
            filler_values.append(-100)
        

    trans_train = replace_nan_values_ordinal_encoding(trans_train, filler_values)
    trans_test = replace_nan_values_ordinal_encoding(trans_test, filler_values)
    X_train[to_encode] = trans_train
    X_test[to_encode] = trans_test

    return X_train, X_test


def apply_nlp_transformation(data:pd.DataFrame, column_name:str, vectorizer) -> pd.DataFrame:
    vectorized_text = vectorizer.transform(data[column_name])
    data = data.drop(columns = [column_name])
    data = data.join(pd.DataFrame(vectorized_text.toarray()))
    return data


def normalize_features(X_train:pd.DataFrame, X_test:pd.DataFrame, exclude_columns:list) -> pd.DataFrame:
    columns = list(X_train.columns)

    no_scale = []

    for one_hot in exclude_columns:
        no_scale_temp =  [col for col in columns if one_hot in str(col)]
        no_scale += no_scale_temp

    to_scale = [col for col in columns if col not in no_scale]

    scaler = StandardScaler()
    scaler.fit(X_train[to_scale])

    X_train[to_scale] = scaler.transform(X_train[to_scale])
    X_test[to_scale] = scaler.transform(X_test[to_scale])

    return X_train, X_test

def process_data(load_dir:str, 
                label_column:str,
                text_column:str,
                drop_duplicates:list,
                drop:list = [],
                quantile_cut:list = [], 
                to_one_hot_encode:list = [],
                to_ordinal_encode:list = [],
                fill_mean:list = [],
                normalize = False,
                fill_nan:float = -1.0,
                nlp_tool:str = None,
                tf_idf_cutoff:float = 0,
                tokenizer = None,
                stop_words:list = stopwords.words('english'),
                token_pattern = "[A-Za-z]+",
                save_dir = None) -> pd.DataFrame:

    data = pd.read_csv(load_dir)
    data = data.drop_duplicates(drop_duplicates)
    data = data.dropna(subset = [label_column])
    

    #TODO: Fill mean by category not by overall mean

    for order in quantile_cut:
        for cat in data[order.per_category].unique():
            quantile = data[data[order.per_category] == cat][order.outlier_column].quantile(order.quantile)
            cat_mean = data[(data[order.per_category] == cat) & (data[order.outlier_column] < quantile)][order.outlier_column].mean()
            data.loc[data[order.per_category] == cat, order.outlier_column] = data[data[order.per_category] == cat][order.outlier_column].fillna(cat_mean)
            
    for to_fill in fill_mean:
        data[to_fill] = data[to_fill].fillna(data[to_fill].mean())

    data = data.drop(columns = drop, errors = "ignore")

    data = data.reset_index(drop = True)

    for to_fill in fill_mean:
        data[to_fill] = data[to_fill].fillna(data[to_fill].mean())

    for column in to_one_hot_encode:
        data = join_one_hot_encoding(data, column)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = [label_column]), data[label_column], test_size=0.20, random_state=0)

    if len(to_ordinal_encode) > 0:
        X_train, X_test = oridnal_encoding(X_train, X_test, to_ordinal_encode)

    if nlp_tool == None:
        X_train = X_train.drop(columns = [text_column])
        X_test = X_test.drop(columns = [text_column])       
    else:
        X_train[text_column] = X_train[text_column].apply(lambda txt: txt.lower())
        X_test[text_column] = X_test[text_column].apply(lambda txt: txt.lower())
        corpus = X_train[text_column]
        
        if nlp_tool == "tf-idf":
            vectorizer = TfidfVectorizer(tokenizer = tokenizer, 
                                        use_idf=True, 
                                        stop_words = stop_words,
                                        min_df = tf_idf_cutoff,
                                        token_pattern = token_pattern)
        if nlp_tool == "bow":
            vectorizer = CountVectorizer(tokenizer = tokenizer, stop_words = stop_words,  token_pattern = token_pattern)

        vectorizer = vectorizer.fit(corpus)

        X_train = apply_nlp_transformation(X_train, text_column, vectorizer)
        X_test = apply_nlp_transformation(X_test, text_column, vectorizer)

        X_train = X_train.fillna(fill_nan)
        X_test = X_test.fillna(fill_nan)

    if normalize:
        X_train, X_test = normalize_features(X_train, X_test, to_one_hot_encode)

    if save_dir != None:

        if save_dir[-1] == "/" or save_dir[-1] == "\\":
            save_dir = save_dir[len(save_dir)-1]

        X_train.join(y_train).to_csv(f"{save_dir}/train_set_{label_column}.csv")
        X_test.join(y_test).to_csv(f"{save_dir}/test_set_{label_column}.csv")

    return X_train, X_test, y_train, y_test


