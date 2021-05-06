import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import(
    TfidfVectorizer,
    CountVectorizer
)

from nltk.corpus import stopwords

#TODO: Add Comments

def join_one_hot_encoding(df:pd.DataFrame, column:str) -> list:
        df = df.join(pd.get_dummies(df[column], prefix=column))
        df = df.drop(columns = [column])
        return df

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
                quantile_cut:dict = {}, 
                to_one_hot_encode:list = [], 
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
    
    for quantile in quantile_cut.keys():
        data = data[data[quantile] < data[quantile].quantile(quantile_cut[quantile])]

    data = data.drop(columns = ["Unnamed: 0", "designation", "province", "region_1", "region_2", "taster_twitter_handle", "title", "variety", "winery"])

    data = data.reset_index(drop = True)

    for to_fill in fill_mean:
        data[to_fill] = data[to_fill].fillna(data[to_fill].mean())

    for column in to_one_hot_encode:
        data = join_one_hot_encoding(data, column)

    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = [label_column]), data[label_column], test_size=0.20, random_state=0)

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


