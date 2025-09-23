import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('mode.chained_assignment', None) 

import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
from scipy.stats import uniform, randint
import numpy as np
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore', category=UserWarning, module='catboost')

RANDOM_SEED = 42

def clean_data(train, test):
    spend_cols = ['RoomService','Spa', 'VRDeck', 'FoodCourt', 'ShoppingMall']
    for df in [train, test]:
        df['Cabin'] = df['Cabin'].fillna('Unknown/-1/U')
        df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
        df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1])
        df['Cabin_side'] = df['Cabin'].apply(lambda x: x.split('/')[2])
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['VIP'] = df['VIP'].fillna(False).astype(bool)
        df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)
        #df['Cabin'] = df['Cabin'].fillna('Unknown')
        df['Destination'] = df['Destination'].fillna('Unknown')
        df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
        df["Group"] = df["PassengerId"].astype(str).str.split("_").str[0]
        df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
        df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
        df["IsChild"] = (df["Age"] < 13).fillna(False).astype(int)
        df["IsTeen"] = ((df["Age"] >= 13) & (df["Age"] < 18)).fillna(False).astype(int)
        df["IsAdult"] = ((df["Age"] >= 18) & (df["Age"] < 65)).fillna(False).astype(int)
        df["IsSenior"] = (df["Age"] >= 65).fillna(False).astype(int)
        df['TotalSpend'] = df[spend_cols].fillna(0).sum(axis=1)
        df['NoSpend'] = (df[spend_cols].fillna(0).sum(axis=1) == 0).astype(int)
        df['Home_Destination'] = df['HomePlanet'] + '_' + df['Destination']

        df['RoomService'] = df['RoomService'].fillna(0)
        df['FoodCourt'] = df['FoodCourt'].fillna(0)
        df['ShoppingMall'] = df['ShoppingMall'].fillna(0)
        df['Spa'] = df['Spa'].fillna(0)
        df['VRDeck'] = df['VRDeck'].fillna(0)

    # Add a marker to split later
    test['Transported'] = None  # Add dummy column to align columns
    combined = pd.concat([train, test], sort=False, ignore_index=True)

    # Encode categorical columns
    le_home = LabelEncoder()
    le_dest = LabelEncoder()
    le_cabin_deck = LabelEncoder()
    le_cabin_side = LabelEncoder()
    le_home_dest = LabelEncoder()

    combined['HomePlanet_enc'] = le_home.fit_transform(combined['HomePlanet'])
    combined['Destination_enc'] = le_dest.fit_transform(combined['Destination'])
    combined['Cabin_deck_enc'] = le_cabin_deck.fit_transform(combined['Cabin_deck'])
    combined['Cabin_side_enc'] = le_cabin_side.fit_transform(combined['Cabin_side'])
    combined['Home_Destination_enc'] = le_home_dest.fit_transform(combined['Home_Destination'])

    # Split back into train and test
    train = combined[combined['Transported'].notnull()].copy()
    test = combined[combined['Transported'].isnull()].copy()

    test.drop(columns=['Transported'], inplace=True)

    return train, test

def cat_feature_selection(X, y, threshold='median', random_state=42, verbose=0):
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.08,
        depth=6,
        random_seed=random_state,
        verbose=verbose
    )

    model.fit(X, y)

    # feature importances
    importances = model.get_feature_importance(type="FeatureImportance")
    feat_names = X.columns
    imp_series = pd.Series(importances, index=feat_names).sort_values(ascending=False)

    # selection
    sfm = SelectFromModel(model, threshold=threshold, prefit=True)
    mask = sfm.get_support()
    selected = feat_names[mask].tolist()

    try:
        scoring = "roc_auc" if len(np.unique(y)) == 2 else "accuracy"
        print(scoring)
        cv_score = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=-1).mean()
    except Exception:
        cv_score = None

    return selected, imp_series, cv_score

def cat_parameter_tuning(X, y, random_state=42, verbose=0):
    base = CatBoostClassifier(verbose=0, random_seed=random_state)

    param_distributions = {
        "iterations": randint(200, 1500),
        "depth": randint(4, 10),
        "learning_rate": uniform(0.005, 0.3),
        "l2_leaf_reg": uniform(0.1, 15),
        "border_count": randint(32, 255),
        "random_strength": uniform(0.0, 3.0),
        "bagging_temperature": uniform(0.0, 1.5),
        "subsample": uniform(0.6, 0.4),
    }

    scoring = "roc_auc" if len(np.unique(y)) == 2 else "accuracy"

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=30,
        scoring=scoring,
        cv=3,
        verbose=verbose,
        random_state=random_state,
        n_jobs=-1,
        return_train_score=False
    )

    search.fit(X, y)
    return search.best_params_, search.best_score_


if __name__ == "__main__":
    train = pd.read_csv('kaggle/input/spaceship-titanic/train.csv')
    test = pd.read_csv('kaggle/input/spaceship-titanic/test.csv')

    train, test = clean_data(train, test)

    train['CryoSleep'] = train['CryoSleep'].astype(int)
    train['VIP'] = train['VIP'].astype(int)

    test['CryoSleep'] = test['CryoSleep'].astype(int)
    test['VIP'] = test['VIP'].astype(int)
    
    features = ['CryoSleep', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend', 'NoSpend', 'HomePlanet_enc', 'Destination_enc', 'Cabin_deck_enc', 'Cabin_side_enc', 'GroupSize', 'IsAlone', 'IsChild', 'IsTeen', 'IsSenior', 'Home_Destination_enc']
    X = train[features]
    y = train['Transported'].astype(int)

    print("CatBoost feature selection:")
    selected_features_cat, imp_series, cv_score = cat_feature_selection(X, y)
    print("Selected features:", selected_features_cat)
    print("Importances:", imp_series)
    print("CV score:", cv_score)

    X_cat_test = test[selected_features_cat]

    print("CatBoost parameter tuning:")
    best_params, best_score = cat_parameter_tuning(X[selected_features_cat], y)
    print("Best parameters:", best_params)
    print("Best score:", best_score)

    cat_model = CatBoostClassifier(verbose=0, random_seed=RANDOM_SEED, **best_params)
    cat_model.fit(X[selected_features_cat], y)
    test['Transported'] = cat_model.predict(X_cat_test)
    test['Transported'] = test['Transported'].astype(bool)
    submission = test[['PassengerId', 'Transported']]
    submission.to_csv('submission_cat.csv', index=False)