import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('mode.chained_assignment', None) 

import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from scipy.stats import uniform, randint
import numpy as np
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='catboost')
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

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

        # Convert boolean columns to integers for lightGBM
        #df['CryoSleep'] = df['CryoSleep'].astype(int)
        #df['VIP'] = df['VIP'].astype(int)

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

def xgb_feature_selection(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select='auto',
        direction='backward',
        scoring='accuracy',
        cv=10,
        n_jobs=-1
    )
    sfs.fit(X_train, y_train)

    selected_features = X_train.columns[sfs.get_support()]
    model.fit(X_train[selected_features], y_train)
    return selected_features, model.score(X_test[selected_features], y_test)

def xgb_parameter_tuning(X, y):
    xgb_param_dist = {
        'n_estimators': [300, 500, 700, 1000],
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0.5, 1, 2, 5]
    }

    xgb_clf = XGBClassifier(
        enable_categorical=True,  # if using categoricals
        tree_method='hist',       # speeds up training, optional
        use_label_encoder=False,  # suppresses warning if using older XGBoost
        eval_metric='logloss',     # or 'error' for accuracy
        random_state=RANDOM_SEED
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=xgb_param_dist,
        n_iter=100,                # increase for a more thorough search
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    xgb_search.fit(X, y)
    return xgb_search.best_params_, xgb_search.best_score_

def lgb_parameter_tuning(X, y):
    #  {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 150, 'num_leaves': 20}
    # 0.8005314351395686
    param_grid = {
        'num_leaves': [5, 20, 31],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
    }
    """param_grid = { # too expensive
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [5, 7, 9],
        'n_estimators': [150, 300, 500],
        'num_leaves': [15, 20, 31, 63],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.01, 0.1, 1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }"""
    lgb_clf = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)
    lgb_search = GridSearchCV(
        estimator=lgb_clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1
    )
    lgb_search.fit(X, y)
    return lgb_search.best_params_, lgb_search.best_score_

def lgb_feature_importance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #lightGBM model fit
    gbm = LGBMClassifier(random_state=42, verbose=-1)
    gbm.fit(X_train, y_train)

    importance_df = pd.DataFrame({'cols':X_train.columns, 'fea_imp':gbm.feature_importances_})
    important_features = importance_df[importance_df['fea_imp'] > 60]['cols'].tolist()

    gbm.fit(X_train[important_features], y_train)

    return important_features, gbm.score(X_test[important_features], y_test)

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
    
    print("XGBoost feature selection:")
    selected_features, test_accuracy = xgb_feature_selection(X, y)
    print("Selected features:", selected_features)
    print("Test accuracy:", test_accuracy)

    print("LightGBM feature importance:")
    feature_importance, test_accuracy = lgb_feature_importance(X, y)
    print("Feature importance:", feature_importance)
    print("Test accuracy:", test_accuracy)

    print("CatBoost feature selection:")
    selected_features_cat, imp_series, cv_score = cat_feature_selection(X, y)
    print("Selected features:", selected_features_cat)
    print("Importances:", imp_series)
    print("CV score:", cv_score)

    X_xgb_test = test[selected_features]
    X_lgb_test = test[feature_importance]
    X_cat_test = test[selected_features_cat]

    print("XGBoost parameter tuning:")
    best_params_xgb, best_score = xgb_parameter_tuning(X[selected_features], y)
    print("Best parameters:", best_params_xgb)
    print("Best score:", best_score)

    print("LightGBM parameter tuning:")
    best_params_lgb, best_score = lgb_parameter_tuning(X[feature_importance], y)
    print("Best parameters:", best_params_lgb)
    print("Best score:", best_score)

    print("CatBoost parameter tuning:")
    best_params_cat, best_score = cat_parameter_tuning(X[selected_features_cat], y)
    print("Best parameters:", best_params_cat)
    print("Best score:", best_score)


    xgb_model = XGBClassifier(**best_params_xgb)
    xgb_model.fit(X[selected_features], y)
    #test['Transported'] = xgb_model.predict(X_xgb_test)
    #test['Transported'] = test['Transported'].astype(bool) # convert back
    #submission = test[['PassengerId', 'Transported']]
    #submission.to_csv('submission_xgb.csv', index=False)

    lgb_model = LGBMClassifier(verbose=0, random_seed=RANDOM_SEED,**best_params_lgb)
    lgb_model.fit(X[feature_importance], y)
    #test['Transported'] = lgb_model.predict(X_lgb_test)
    #test['Transported'] = test['Transported'].astype(bool) # convert back
    #submission = test[['PassengerId', 'Transported']]
    #submission.to_csv('submission_lgb.csv', index=False)

    cat_model = CatBoostClassifier(verbose=0, random_seed=RANDOM_SEED, **best_params_cat)
    cat_model.fit(X[selected_features_cat], y)

    # Prepare the stacking ensemble
    stack = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            #('lgb', lgb_model),
            ('cat', cat_model)
            # You can add CatBoost or others here if you want
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1,
        passthrough=False
    )

    all_features = list(set(selected_features) | set(feature_importance))
    X_stack = X[all_features]
    X_test_stack = test[all_features]

    # Fit the stacking ensemble
    stack.fit(X_stack, y)

    # Predict on test set
    test['Transported'] = stack.predict(X_test_stack)
    test['Transported'] = test['Transported'].astype(bool)
    submission = test[['PassengerId', 'Transported']]
    submission.to_csv('submission_stack.csv', index=False)


    

    



