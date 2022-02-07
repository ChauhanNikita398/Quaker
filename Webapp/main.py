from flask import render_template, flash, request
import logging, io, base64, os, datetime
from datetime import datetime
from datetime import timedelta
import xgboost as xgb
from Webapp import app
import pickle
# import pickle5 as pickle
import numpy as np # linear algebra
import pandas as pd # dataframes
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer,  RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import csv
from pathlib import Path

# global variables
earthquake_live = None
days_out_to_predict = 7


#app = Flask(__name__)

def prepare_earthquake_data_and_model(days_out_to_predict = 7, max_depth=3, eta=0.1):
    
    # get latest data from USGS servers
    df = pd.read_csv('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv')
    df = df.sort_values('time', ascending=True)
    # truncate time from datetime
    df['date'] = df['time'].str[0:10]

    # only keep the columns needed
    df = df[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]
    temp_df = df['place'].str.split(', ', expand=True) 
    df['place'] = temp_df[1]
    df = df[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]

    # calculate mean lat lon for simplified locations
    df_coords = df[['place', 'latitude', 'longitude']]
    df_coords = df_coords.groupby(['place'], as_index=False).mean()
    df_coords = df_coords[['place', 'latitude', 'longitude']]

    df = df[['date', 'depth', 'mag', 'place']]
    df = pd.merge(left=df, right=df_coords, how='inner', on=['place'])

    # loop through each zone and apply MA
    eq_data = []
    df_live = []
    for symbol in list(set(df['place'])):
        temp_df = df[df['place'] == symbol].copy()
        temp_df['depth_avg_22'] = temp_df['depth'].rolling(window=22,center=False).mean() 
        temp_df['depth_avg_15'] = temp_df['depth'].rolling(window=15,center=False).mean()
        temp_df['depth_avg_7'] = temp_df['depth'].rolling(window=7,center=False).mean()
        temp_df['mag_avg_22'] = temp_df['mag'].rolling(window=22,center=False).mean() 
        temp_df['mag_avg_15'] = temp_df['mag'].rolling(window=15,center=False).mean()
        temp_df['mag_avg_7'] = temp_df['mag'].rolling(window=7,center=False).mean()
        temp_df.loc[:, 'mag_outcome'] = temp_df.loc[:, 'mag_avg_7'].shift(days_out_to_predict * -1)

        df_live.append(temp_df.tail(days_out_to_predict))

        eq_data.append(temp_df)

    # concat all location-based dataframes into master dataframe
    df = pd.concat(eq_data)

    # remove any NaN fields
    df = df[np.isfinite(df['depth_avg_22'])]
    df = df[np.isfinite(df['mag_avg_22'])]
    df = df[np.isfinite(df['mag_outcome'])]

    # prepare outcome variable
    df['mag_outcome'] = np.where(df['mag_outcome'] > 2.5, 1,0)

    df = df[['date',
             'latitude',
             'longitude',
             'depth_avg_22',
             'depth_avg_15',
             'depth_avg_7',
             'mag_avg_22', 
             'mag_avg_15',
             'mag_avg_7',
             'mag_outcome']]

    # keep only data where we can make predictions
    df_live = pd.concat(df_live)
    df_live = df_live[np.isfinite(df_live['mag_avg_22'])]

    # let's train the model whenever the webserver is restarted
    features = [f for f in list(df) if f not in ['date', 'mag_outcome', 'latitude',
     'longitude']]

    X_train, X_test, y_train, y_test = train_test_split(df[features],
                         df['mag_outcome'], test_size=0.3, random_state=42)

    dtrain = xgb.DMatrix(X_train[features], label=y_train)
    dtest = xgb.DMatrix(X_test[features], label=y_test)

    param = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'eval_metric': 'auc',
            'min_child_weight': 4,
            'max_depth': 5,  # the maximum depth of each tree
            'eta': 0.1,  # the training step for each iteration
            'gamma':0.2
            }  
    num_round = 1000  # the number of training iterations 
    xgb_model = xgb.train(param, dtrain, num_round) 


    # train on live data
    dlive = xgb.DMatrix(df_live[features])  
    preds = xgb_model.predict(dlive)

    # add preds to live data
    df_live = df_live[['date', 'place', 'latitude', 'longitude']]
    # add predictions back to dataset 
    df_live = df_live.assign(preds=pd.Series(preds).values)

    # aggregate down dups
    df_live = df_live.groupby(['date', 'place'], as_index=False).mean()

    # increment date to include DAYS_OUT_TO_PREDICT
    df_live['date']= pd.to_datetime(df_live['date'],format='%Y-%m-%d') 
    df_live['date'] = df_live['date'] + pd.to_timedelta(days_out_to_predict,unit='d')

    return(df_live)

def get_earth_quake_estimates(desired_date, df_live):
    
    from datetime import datetime
    live_set_tmp = df_live[df_live['date'] == desired_date]

    # format lat/lons like Google Maps expects
    LatLngString = ''
    if (len(live_set_tmp) > 0):
        for lat, lon, pred in zip(live_set_tmp['latitude'], live_set_tmp['longitude'], live_set_tmp['preds']): 
            # this is the threashold of probability to decide what to show and what not to show
            if (pred > 0.3):
                LatLngString += "new google.maps.LatLng(" + str(lat) + "," + str(lon) + "),"

    return(LatLngString)


@app.before_first_request
def startup():
    global earthquake_live

    # prepare earthquake data, model and get live data set with earthquake forecasts
    earthquake_live = prepare_earthquake_data_and_model()


@app.route("/prevention", methods=['POST', 'GET'])
def build_page():
        if request.method == 'POST':

            horizon_int = int(request.form.get('slider_date_horizon'))
            horizon_date = datetime.today() + timedelta(days=horizon_int)

            return render_template('index.html',
                date_horizon = horizon_date.strftime('%m/%d/%Y'),
                earthquake_horizon = get_earth_quake_estimates(str(horizon_date)[:10], earthquake_live),
                current_value=horizon_int, 
                days_out_to_predict=days_out_to_predict)

        else:
            # set blank map
            return render_template('index.html',
                date_horizon = datetime.today().strftime('%m/%d/%Y'),
                earthquake_horizon = '',
                current_value=0,
                days_out_to_predict=days_out_to_predict)


@app.route('/')
def hello_world():
    return render_template("home.html")


def prepare_csv(features):
    f1=[1211,31,22,242]
    f2=[1212,31,22,242]
    f3=[1213,31,22,242]
    f4=[1214,31,22,242]
    
    for i in range(0,13):
        f1.append(features[i])
        f2.append(features[i])
        f3.append(features[i])
        f4.append(features[i])
    
    if(features[13]=="has_superstructure_adobe_mud"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)

    if(features[13]=="has_superstructure_mud_mortar_stone"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)
    
    if(features[13]=="has_superstructure_stone_flag"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)
    
    if(features[13]=="has_superstructure_cement_mortar_stone"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)

    if(features[13]=="has_superstructure_mud_mortar_brick"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)

    if(features[13]=="has_superstructure_cement_mortar_brick"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)

    if(features[13]=="has_superstructure_timber"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)

    if(features[13]=="has_superstructure_bamboo"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)

    if(features[13]=="has_superstructure_rc_non_engineered"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)

    if(features[13]=="has_superstructure_rc_engineered"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)

    if(features[13]=="has_superstructure_other"):
        f1.append(1)
        f2.append(1)
        f3.append(1)
        f4.append(1)
    else:
        f1.append(0)
        f2.append(0)
        f3.append(0)
        f4.append(0)
    
    f1.append(features[14])
    f2.append(features[14])
    f3.append(features[14])
    f4.append(features[14])

    f1.append("null")
    f2.append("null")
    f3.append("null")
    f4.append("null")

    f1.append("Reconstruction")
    f2.append("Major repair")
    f3.append("Minor repair")
    f4.append("No need")
    
    data = [f1,f2,f3,f4]
    with open('Webapp\Values.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write multiple rows
        writer.writerows(data)

    
    return 1
    



# Convert to object Transformer
def covert_to_object(x):
    '''Converts a column to object'''
    return pd.DataFrame(x).astype(object)

def prepare_output(model,df):
    df_stru = pd.read_csv("Webapp\Values.csv", index_col = 'building_id')

    # Convert data types to categorical
    df = df.astype({'district_id': 'object', 'vdcmun_id': 'object', 'ward_id': 'object'})
    df_stru = df_stru.astype({'district_id': 'object', 'vdcmun_id': 'object', 'ward_id': 'object'})

    # Drop Rows with missing data
    df.dropna(inplace = True)
    # df_stru.dropna(inplace = True)
    
    # ----------FEATURE ENGINEERING--------------
    # New fields to add
    df['net_rooms'] = df.count_floors_post_eq - df.count_floors_pre_eq
    df['net_height'] = df.height_ft_post_eq - df.height_ft_pre_eq
    df_stru['net_rooms'] = df_stru.count_floors_post_eq - df_stru.count_floors_pre_eq
    df_stru['net_height'] = df_stru.height_ft_post_eq - df_stru.height_ft_pre_eq

    # Create training and testing data
    x_train= df.drop('damage_grade',axis=1)
    x_test= df_stru.drop('damage_grade', axis = 1)
    print(x_train.shape)
    print(x_test.shape)

    # ------------ Predictor Processing ------------

    # Identify columns
    fts_cvt_obj = ['district_id', 'vdcmun_id', 'ward_id']
    fts_outlier = ['age_building']
    fts_cat = df_stru.drop(fts_cvt_obj, axis = 1).select_dtypes(include=['object']).columns #.drop('damage_grade', axis = 1)
    fts_num = df_stru.select_dtypes(np.number).drop('damage_grade', axis = 1).columns

    # print(fts_cvt_obj)
    # print(fts_outlier)
    # print(fts_cat)
    # print(fts_num)
    trans_to_object = Pipeline(steps = [('convert_to_object', FunctionTransformer(covert_to_object))])
    # Outlier Restriction
    trans_outlier = Pipeline(steps = [('Outlier_scaler', RobustScaler(quantile_range = (0,0.9)))])

    # Categorical Transformer
    trans_cat = Pipeline(steps = [('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # # Numerical Transformer
    trans_num = Pipeline(steps = [('scaler', StandardScaler()), ('MinMax', MinMaxScaler())])

    # Zero or Near Zero variance
    trans_nzv = Pipeline(steps = [('nzv', VarianceThreshold(threshold = 0.01))])

    # Create a single Preprocessing step for predictors
    preprocessor_preds = ColumnTransformer(
        transformers=[
            ('convert_to_object', trans_to_object, fts_cvt_obj), # Convert data types
            ('Outlier', trans_outlier, fts_outlier), # Outlier treatment 
            ('num', trans_num, fts_num), # Centre and scale
            ('cat', trans_cat, fts_cat), # One Hot encode
            ('nzv', trans_nzv,[]) # One Hot encode
        ])
      
    # print(x_train.head())
    # Apply the transformations to both train and test
    x_train = preprocessor_preds.fit_transform(x_train)
    print(x_train.shape)
    x_test = preprocessor_preds.transform(x_test)
    final_y_pred = model.predict(x_test)
    final_y=np.ceil(final_y_pred)

    return final_y


def clear_csv():
    file = open('Webapp\Values.csv', 'r+')
    file.truncate(0) 
    # data = pd.read_csv("Webapp\Initial.csv")
    # data =data.columns
    # print(data)
    list= [['building_id', 'district_id', 'vdcmun_id', 'ward_id','count_floors_pre_eq', 'count_floors_post_eq', 'age_building',
       'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq','land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position','plan_configuration', 'has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag','has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick','has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered','has_superstructure_rc_engineered', 'has_superstructure_other',
       'condition_post_eq', 'damage_grade', 'technical_solution_proposed']]
    with open('Webapp\Values.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write multiple rows
        writer.writerows(list)

    return 1

@app.route('/recovery',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        model1=pickle.load(open('Webapp\model.pkl','rb'))
        
        df_stru = pd.read_csv("Webapp\csv_building_structure.csv",index_col = 'building_id')
        features=[x for x in request.form.values()]
        prepare_csv(features)
        output=prepare_output(model1,df_stru)
        return render_template('result.html',pred=output)
    else:
        # set blank map
        clear_csv()
        return render_template('building.html')
