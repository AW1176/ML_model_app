import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

# function for converting mol obj into feature matrix of descriptors
@st.cache_data 
def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule

        missingVal is used if the descriptor cannot be calculated
    '''
    dfs = []
    res = {}
    df = pd.Series()
    merged_df = pd.DataFrame()

    for name,func in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = func(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[name] = val
    df = pd.Series(res)
    #dfs.append(df)
    #merged_df = pd.concat(dfs, axis=1)
    return df

@st.cache_resource
def df_to_features(mols_df):
    total_mols = len(mols_df)
    #progress_bar = st.progress(0)
    feature_matrix = []
    for i, mol in enumerate(mols_df['rdkit_mol']):
        descriptors = getMolDescriptors(mol)
        feature_matrix.append(descriptors)
        
        # Update progress bar
    #    progress = (i+1) / total_mols
    #    progress_bar.progress(progress,f"Converted {int(progress * 100)}%")
    #progress_bar.empty()
    feature_matrix = pd.concat(feature_matrix, axis=1).T
    feature_matrix.index = mols_df.SMILES
    return feature_matrix

# Function for converting columns of dataset to X & Y
@st.cache_resource
def column_to_XY_df(dframe,X_col_nme,y_col_nme):
    if X_col_nme and y_col_nme is not None:
        smiles_df = dframe[[X_col_nme]].copy()
        target_df = dframe[[y_col_nme]].copy()
        smiles_df['rdkit_mol'] = smiles_df[X_col_nme].apply(smile_to_mol)
        return smiles_df, target_df
    else:
        return None,None

@st.cache_data
def evaluate(_model, test_features, test_labels):
    predictions = _model.predict(test_features)
    mae = mean_absolute_error(test_labels, predictions)
    accuracy = 100 - (100 * (mae / np.mean(test_labels)))
    return accuracy,mae

@st.cache_data
def single_smiles_features(smiles):
    mol = smile_to_mol(smiles)
    return getMolDescriptors(mol=mol)


# Function for converting SMILES into mol 
@st.cache_data
def smile_to_mol(smile):
  mol = Chem.MolFromSmiles(smile)
  mol = Chem.AddHs(mol)
  return mol

# function for fitting model and getting results
def fit_model(X,y,models,param_grid,cv):
    results = []
    for name, model in models.items():
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid[name],
            return_train_score=True,
            cv=cv).fit(X, y)
        result = {"model": name, "cv_results": pd.DataFrame(grid_search.cv_results_)}
        results.append(result)
    return results

# function for calling sample
@st.cache_data
def call_df(uploaded_file = '',load_sample = False):
    df = pd.DataFrame()  # Define a default value for df
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file.name)
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file.name)
        return df
    except:
        if load_sample:
            df = pd.read_excel("solubility_data.xlsx")
            st.write('loading solubility data')
            return df

models = {
    "Random Forest": RandomForestRegressor(
        min_samples_leaf=5, random_state=0, n_jobs=4
    ),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_leaf_nodes=15, random_state=0, early_stopping=False
    ),
}
param_grids = {
    "Random Forest": {"n_estimators": [10, 20, 50, 100]},
    "Hist Gradient Boosting": {"max_iter": [10, 20, 50, 100, 300, 500]},
}