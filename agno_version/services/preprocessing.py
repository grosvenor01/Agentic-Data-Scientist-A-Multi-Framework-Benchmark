import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def scale_standard(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def scale_minmax(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def scale_maxabs(df, columns):
    scaler = MaxAbsScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def scale_robust(df, columns):
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def normalize(df, columns, norm='l2'):
    normalizer = Normalizer(norm=norm)
    df[columns] = normalizer.fit_transform(df[columns])
    return df



def encode_onehot(df, columns):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
    df = df.drop(columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def encode_ordinal(df, columns):
    encoder = OrdinalEncoder()
    df[columns] = encoder.fit_transform(df[columns])
    return df

def encode_label(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df

def poly_features(df, columns, degree=2, include_bias=False):
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_array = poly.fit_transform(df[columns])
    poly_df = pd.DataFrame(poly_array, columns=poly.get_feature_names_out(columns))
    df = df.drop(columns, axis=1)
    df = pd.concat([df, poly_df], axis=1)
    return df



def impute_simple(df, columns, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df

def impute_knn(df, columns, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[columns] = imputer.fit_transform(df[columns])
    return df

def transform_power(df, columns, method='yeo-johnson'):
    pt = PowerTransformer(method=method)
    df[columns] = pt.fit_transform(df[columns])
    return df

def transform_quantile(df, columns, output_distribution='uniform'):
    qt = QuantileTransformer(output_distribution=output_distribution)
    df[columns] = qt.fit_transform(df[columns])
    return df


def remove_low_variance(df, threshold=0.0):
    sel = VarianceThreshold(threshold=threshold)
    df_sel = sel.fit_transform(df)
    return pd.DataFrame(df_sel, columns=df.columns[sel.get_support()])

def select_kbest(df, y, k=10):
    sel = SelectKBest(score_func=f_classif, k=k)
    df_sel = sel.fit_transform(df, y)
    return pd.DataFrame(df_sel, columns=df.columns[sel.get_support()])

def reduce_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)
    cols = [f'PC{i+1}' for i in range(n_components)]
    return pd.DataFrame(df_pca, columns=cols)



def build_column_transformer(numeric_cols=None, categorical_cols=None, 
                             numeric_scaler=StandardScaler(), categorical_encoder=OneHotEncoder(sparse=False)):
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_scaler, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_encoder, categorical_cols))
    return ColumnTransformer(transformers)
