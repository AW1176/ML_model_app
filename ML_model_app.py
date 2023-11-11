import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import utils
from joblib import dump, load

st.write("# Welcome to App: Use SMILES data to predict target")


cv = KFold(n_splits=4, shuffle=True, random_state=0)



def smiles_col_nme_callback():
    if 'smiles_col_nme' in st.session_state: st.session_state.smiles_col_nme = smiles_col_nme
def target_col_nme_callback():
    if 'target_col_nme' in st.session_state: st.session_state.target_col_nme = target_col_nme


@st.cache_data
def get_feature_matrix():
    smiles_df, target_df = utils.column_to_XY_df(dframe=df,X_col_nme=smiles_col_nme,y_col_nme=target_col_nme)
    feature_matrix = utils.df_to_features(mols_df=smiles_df)
    if 'feature_matrix' in st.session_state: st.session_state.feature_matrix = feature_matrix
    if 'target_df' in st.session_state: st.session_state.target_df = target_df
    if 'smiles_df' in st.session_state: st.session_state.smiles_df = smiles_df
    return smiles_df,target_df,feature_matrix

#_______________________

#read data using pandas

#_______________________

if 'smiles_col_nme' not in st.session_state:st.session_state.smiles_col_nme=None
if 'target_col_nme' not in st.session_state:st.session_state.target_col_nme=None
if 'df' not in st.session_state: st.session_state.df = pd.DataFrame()
if 'smiles_df' not in st.session_state: st.session_state.smiles_df = pd.DataFrame()
if 'target_df' not in st.session_state: st.session_state.target_df = pd.DataFrame()
if 'feature_matrix' not in st.session_state: st.session_state.feature_matrix = pd.DataFrame()
if 'user_smile' not in st.session_state: st.session_state.user_smile = None
if 'start_training' not in st.session_state: st.session_state.start_training = False
if 'best_parameter' not in st.session_state: st.session_state.best_parameter = None
if 'best_grid' not in st.session_state: st.session_state.best_grid = None

smiles_df, target_df, feature_matrix = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()

uploaded_file = st.file_uploader("Choose a file",help="csv or xlsx are supported only")
load_sample = st.checkbox("Load sample data?")


#st.write(call_df(uploaded_file=uploaded_file,load_sample=load_sample))
df = utils.call_df(uploaded_file=uploaded_file,load_sample=load_sample)
if 'df' in st.session_state: st.session_state.df = df

if df is not None:
    st.dataframe(df, use_container_width=True)
    st.markdown("##### Data uploaded must contain a SMILES and target column")
    
col1,col2 = st.columns([1,1])
with col1:
    smiles_col_nme = st.text_input("Enter name of SMILES column","SMILES",
                                    help="Enter name of column containing SMILES",
                                    on_change=smiles_col_nme_callback)
with col2:
    target_col_nme = st.text_input("Enter name of target column",
                                    "Solubility",on_change=target_col_nme_callback,
                                    help="Enter name of column containing target")           

if smiles_col_nme and target_col_nme is not None:
    if df is not None:
        smiles_df,target_df,feature_matrix = get_feature_matrix()
        
#feature_matrix = df_to_features(mols_df=smiles_df)

columns_name = st.session_state.feature_matrix.columns
with st.expander("See feature matrix"):
    st.dataframe(st.session_state.feature_matrix)

model_availables = ['RandomForest']# add more model and implement logic 

st.write('# Develop ML Model now !')
col3,col4 = st.columns([1,1])
col5 = st.columns([1])[0]
col6 = st.columns([1])[0]

with col3:
    model_selected = st.selectbox('please select a model',model_availables)

    if model_selected=='RandomForest':
        st.markdown(f"You selected **{model_selected}**")
    elif model_selected=='LinearRegression':
        st.write(f"You selected **{model_selected}**")

with col4:
    st.write('Hyperparameters are for RandomForest only')
    custom_parm = st.checkbox('Do you want to select hyperparameter for GridSearch')
    default_parm = st.checkbox('Do you want to use default hyperparameter for GridSearch')

with col5:
    if custom_parm:
        col5a,col5b,col5c = st.columns([1,1,1])
        if model_selected=='RandomForest':
            with col5a:
                n_estimators = st.multiselect('Select n_estimators',[10,100,500,1000])
                #st.write(n_estimators)
            with col5b:
                bootstrap = st.multiselect('Bootstrap?',[True,False])
                #st.write(bootstrap)
            with col5c:
                max_depth = st.multiselect('max_depth',[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None])
                #st.write(max_depth)

            with col6:
                col6a,col6b,col6c = st.columns([1,1,1])
                with col6a:
                    min_samples_split = st.multiselect('min_samples_split',[2, 5, 10])
                    #st.write(min_samples_split)
                with col6b:
                    min_samples_leaf = st.multiselect('min_samples_leaf',[1, 2, 4])
                    #st.write(min_samples_leaf)
                with col6c:
                    max_features = st.multiselect('max_features',['log2', 'sqrt'])
                    #st.write(max_features)
    elif default_parm:
        col5a,col5b,col5c = st.columns([1,1,1])
        if model_selected=='RandomForest':
            with col5a:
                n_estimators = [10,100,500,1000]
                st.write(f"n_estimators = {n_estimators}")
            with col5b:
                bootstrap = [True,False]
                st.write(f"bootstrap = {bootstrap}")
            with col5c:
                max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
                st.write(f"max_depth = {max_depth}")
            with col6:
                col6a,col6b,col6c = st.columns([1,1,1])
                with col6a:
                    min_samples_split = [2, 5, 10]
                    st.write(f"min_samples_split = {min_samples_split}")
                with col6b:
                    min_samples_leaf = [1, 2, 4]
                    st.write(f"min_samples_leaf = {min_samples_split}")
                with col6c:
                    max_features = ['auto', 'sqrt']
                    st.write(f"max_features = {max_features}")
col7 = st.columns(1)[0]

try:
    parm_dict = {
        'n_estimators' : n_estimators,
        'bootstrap' : bootstrap,
        'max_depth' : max_depth,
        'min_samples_split' : min_samples_split,
        'min_samples_leaf' : min_samples_leaf,
        'max_features' : max_features
    }
except:
    st.write('')

@st.cache_resource
def train_model(_model,X,y):
    _model.fit(X,y)
    best_grid = grid_search.best_estimator_ # select best estimator 
    best_parameter = grid_search.best_params_
    #grid_accuracy,grid_mae = evaluate(best_grid, X_test, y_test)
    
    return best_grid,best_parameter

def store_smile():
    if 'user_smile' in st.session_state:st.session_state.user_smile = user_smiles

def training_session():
    if 'start_training' in st.session_state: st.session_state.start_training = True

X_features = np.array(st.session_state.feature_matrix)
y_target = np.array(st.session_state.target_df)
#st.dataframe

X_train,X_test,y_train,y_test = train_test_split(X_features,y_target,test_size=0.5,random_state=42)

feature_importaces,importtant = None,None # best_grid,best_parameter,
columns_name_series,feature_importaces_series,sorted_idx = None,None,None

with col7:
    st.write('# Train your model now ?')
    start_training = st.button('Start Training',on_click=training_session)
    if st.session_state.start_training:
        if model_selected=='RandomForest':
            rf = RandomForestRegressor(random_state = 42)
            grid_search = GridSearchCV(estimator = rf, param_grid = parm_dict,cv = 3, n_jobs = -1,refit=True)
            best_grid,best_parameter = train_model(grid_search,X_train,y_train)
            #if 'best_grid' in st.session_state.best_grid: st.session_state.best_grid = best_grid
            #if 'best_parameter' in st.session_state.best_parameter: st.session_state.best_parameter = best_parameter
            
            y_pred_train = best_grid.predict(X_train)
            y_pred_test = best_grid.predict(X_test)
            
            accuracy_train = r2_score(y_train, y_pred_train)
            accuracy_test = r2_score(y_test, y_pred_test)
            #grid_search.fit(X_train,y_train)
            
            mae_test = mean_absolute_error(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            
            mse_train = mean_squared_error(y_train,y_pred_train)
            mse_test = mean_squared_error(y_test,y_pred_test)
            
            model_performance_dict = {'r2_train':accuracy_train,'r2_test':accuracy_test,'MAE_train':mae_train,'MAE_test':mae_test,'MSE_train':mse_train,'MSE_test':mse_test}
            model_performance_df = pd.Series(model_performance_dict)
            best_parameter_df = pd.Series(best_parameter)
            
            feature_importaces = best_grid.feature_importances_
            feature_importaces = np.array(feature_importaces)
            col7a,col7b,col7c = st.columns([1,1,1])
            
            with col7a:
                st.write('Best parameter obtained are:')
                st.dataframe(best_parameter_df)
            with col7b:
                st.write('Model Performance:')
                st.dataframe(model_performance_df)

            with col7c:
                st.write('Top 10 important Features')
                print(feature_importaces)
                sorted_idx = best_grid.feature_importances_.argsort()[::-1][:10]
                #feature_importaces[sorted_idx]
                #st.dataframe(columns_name[sorted_idx])
                columns_name_series = pd.Series(columns_name[sorted_idx],name='descriptors')
                feature_importaces_series = pd.Series(feature_importaces[sorted_idx],name='importances')
                importtant = pd.concat([columns_name_series,feature_importaces_series],axis=1)
                st.dataframe(importtant,use_container_width=True)


            
            #fig, ax = plt.subplots()
            #ax.bar(np.array(columns_name[sorted_idx]), np.array(feature_importaces[sorted_idx]))
            # Add labels and title
            #ax.set_xlabel('Features')
            #ax.set_title('Features importance of descriptors')
            fig, ax = plt.subplots()
            ax.barh(importtant["descriptors"], importtant["importances"])
            ax.set_xlabel('Feature Importances')
            with st.expander("See feature importances"):
                st.pyplot(fig)

col8 = st.columns(1)[0]
with col8:
    st.write('# Do you want to Predict target?')
    user_smiles = st.text_input('Enter your SMILES here',on_change=store_smile)
    col8a,col8b,col8c = st.columns([1,1,1])
    if user_smiles:
        featurized_smile = utils.single_smiles_features(smiles=user_smiles)
        with col8a:
            st.dataframe(featurized_smile)
        with col8b:
            user_mol = Chem.MolFromSmiles(user_smiles)
            user_img = Draw.MolToImage(user_mol)
            st.image(user_img)
        with col8c:
            featurized_smile = np.array(featurized_smile).reshape(1, -1)
            st.write('#### The predicted solubility of provided smile is: ')
            user_predicted_smile = best_grid.predict(featurized_smile)
            st.write(user_predicted_smile)

            if st.checkbox('Happy to Export model ?'):
                dump(best_grid, "best_model.joblib")
                st.write('Model Exported in CWD')




