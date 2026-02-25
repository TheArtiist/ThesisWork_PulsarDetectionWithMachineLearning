from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import (
    RandomUnderSampler,
    TomekLinks,
    ClusterCentroids,
    NearMiss,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    OneSidedSelection
)

def apply_random_oversampling(X, y, random_state=42):
    ros = RandomOverSampler(random_state=random_state)
    return ros.fit_resample(X, y)

def apply_smote(X, y, random_state=42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X, y)

def apply_borderline_smote(X, y, random_state=42):
    bsmote = BorderlineSMOTE(random_state=random_state)
    return bsmote.fit_resample(X, y)

def apply_adasyn(X, y, random_state=42):
    adasyn = ADASYN(random_state=random_state)
    return adasyn.fit_resample(X, y)

def apply_tomek_links(X, y):
    tl = TomekLinks()
    return tl.fit_resample(X, y)

def apply_cluster_centroids(X, y, random_state=42):
    cc = ClusterCentroids(random_state=random_state)
    return cc.fit_resample(X, y)

def apply_random_undersampling(X, y, random_state=42):
    rus = RandomUnderSampler(random_state=random_state)
    return rus.fit_resample(X, y)

def apply_nearmiss(X, y, version=1):
    nm = NearMiss(version=version)
    return nm.fit_resample(X, y)

def apply_condensed_nearest_neighbor(X, y, random_state=42):
    cnn = CondensedNearestNeighbour(random_state=random_state)
    return cnn.fit_resample(X, y)

def apply_edited_nearest_neighbor(X, y):
    enn = EditedNearestNeighbours()
    return enn.fit_resample(X, y)

def apply_one_sided_selection(X, y, random_state=42):
    oss = OneSidedSelection(random_state=random_state)
    return oss.fit_resample(X, y)

def get_resampled_data(method, X, y, **kwargs):
    """
    Wrapper function to quickly test different resampling techniques.
    
    Supported methods:
    'random_oversampling', 'smote', 'borderline_smote', 'adasyn', 
    'tomek_links', 'cluster_centroids', 'random_undersampling', 
    'nearmiss', 'condensed_nearest_neighbor', 'edited_nearest_neighbor', 
    'one_sided_selection'
    """
    resamplers = {
        'random_oversampling': apply_random_oversampling,
        'smote': apply_smote,
        'borderline_smote': apply_borderline_smote,
        'adasyn': apply_adasyn,
        'tomek_links': apply_tomek_links,
        'cluster_centroids': apply_cluster_centroids,
        'random_undersampling': apply_random_undersampling,
        'nearmiss': apply_nearmiss,
        'condensed_nearest_neighbor': apply_condensed_nearest_neighbor,
        'edited_nearest_neighbor': apply_edited_nearest_neighbor,
        'one_sided_selection': apply_one_sided_selection
    }
    
    if method not in resamplers:
        raise ValueError(f"Unknown sampling method: {method}")
        
    return resamplers[method](X, y, **kwargs)

if __name__ == '__main__':
    import pandas as pd
    import time
    from collections import Counter
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    print("--- Imbalanced Data Resampling with HTRU2 Dataset ---")

    try:
        train_df = pd.read_csv("htru2/train.csv")
        test_df = pd.read_csv("htru2/validation.csv")
    except FileNotFoundError:
        print("Hiba: 'htru2/train.csv' vagy 'htru2/validation.csv' nem található.")
        exit(1)

    target_column = "label"

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    print(f"Original train dataset shape: {Counter(y_train)}\n")

    methods_to_test = [
        'original (no resampling)',
        'random_oversampling', 'smote', 'borderline_smote', 'adasyn', 
        'tomek_links', 'cluster_centroids', 'random_undersampling', 
        'nearmiss', 'condensed_nearest_neighbor', 'edited_nearest_neighbor', 
        'one_sided_selection'
    ]

    for method in methods_to_test:
        try:
            print(f"\n[{method.upper()}]")
            
            if method == 'original (no resampling)':
                X_train_res = X_train
                y_train_res = y_train
            else:
                start_time = time.time()
                X_train_res, y_train_res = get_resampled_data(method, X_train, y_train)
                resampling_time = time.time() - start_time
                print(f" Resampled shape: {Counter(y_train_res)} | Resampling time: {resampling_time:.3f}s")
            
            clf = RandomForestClassifier(random_state=42, n_jobs=-1)
            clf.fit(X_train_res, y_train_res)
            
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            print(f" Recall for class 1: {report['1']['recall']:.4f}")
            print(f" F1-score for class 1: {report['1']['f1-score']:.4f}")
            print(f" Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")

        except Exception as e:
            print(f" Hiba történt: {e}")

