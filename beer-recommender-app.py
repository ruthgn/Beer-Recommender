import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

BEER_EMOJI_URL = "https://img.charactermap.one/google/android-11/512px/1f37a.png"

# Set page title and favicon.
st.set_page_config(
    page_title="Beer Recommender", page_icon=BEER_EMOJI_URL,
)

########## Main Panel

# Header and Description
st.write("""
# How 'bout a pint? 🍺

This app will generate two sets of beer recommendations for you—the first set will list beers of your **chosen style**, 
while the second one displays recommended beers of **other styles**.

All recommended beers are selected based on their similarities with your beer of choice and ranked based on their overall consumer review scores.
""")

##########


######### Sidebar
st.sidebar.header('Beer Recommender 🔮')
# Caption
st.sidebar.caption("by [Ruth G. N.](https://www.linkedin.com/in/ruthgn/)")

# Social Links
st.sidebar.write("""
[![Follow](https://img.shields.io/twitter/follow/RuthInData?style=social)](https://www.twitter.com/RuthInData)
&nbsp[![Fork](https://img.shields.io/github/forks/ruthgn/Beer-Recommender.svg?logo=github&style=social)](https://github.com/ruthgn/Beer-Recommender)
"""
)

st.sidebar.markdown("Just in case you need help deciding what beer to drink next. *Cheers!*🍻")
st.sidebar.markdown("----")

st.sidebar.header('Data')
st.sidebar.markdown("""
Recommendations are generated from a list of 3197 unique beers from 934 different breweries, while taking into account the following aspects of each beer:

* Alcohol content (% by volume)
* Minimum and maximum IBU (International Bitterness Units)
* Mouthfeel:
    * Astringency
    * Body
    * Alcohol
* Taste:
    * Bitter
    * Sweet
    * Sour
    * Salty
* Flavor And Aroma:
    * Fruits
    * Hoppy
    * Spices
    * Malty 
""")
st.sidebar.markdown("----")

st.sidebar.header('About The Recommendation Engine')
st.sidebar.markdown("""
Kaggle notebook outlining the recommendation engine building process is available [here](https://www.kaggle.com/ruthgn/what-we-talk-about-when-we-talk-about-good-beer).
""")

##########


########## Main Panel

# Setting input parameters (collecting user input)

st.markdown("----")
st.markdown("\n")
st.markdown("\n")

##########


# Data Preprocessing

full_data = pd.read_csv('updated_beer_profile_and_ratings.csv')

# List numeric features (columns) of different types
tasting_profile_cols = ['Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']
chem_cols = ['ABV', 'Min IBU', 'Max IBU']

# Scaling data
def scale_col_by_row(df, cols):
    scaler = MinMaxScaler()
    # Scale values by row
    scaled_cols = pd.DataFrame(scaler.fit_transform(df[cols].T).T, columns=cols)
    df[cols] = scaled_cols
    return df

def scale_col_by_col(df, cols):
    scaler = MinMaxScaler()
    # Scale values by column
    scaled_cols = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)
    df[cols] = scaled_cols
    return df

# Scale values in tasting profile features (across rows)
data = scale_col_by_row(full_data, tasting_profile_cols)

# Scale values in tasting profile features (across columns)
data = scale_col_by_col(full_data, tasting_profile_cols)

# Scale values in chemical features (across columns)
data = scale_col_by_col(full_data, chem_cols)


# Use only numeric features for determining nearest neighbors
df_num = data.select_dtypes(exclude=['object'])


########## Main Panel

# User Input

def user_input_features():
    Style = st.selectbox("What's your favorite beer style?", (data['Style'].unique()))

    style_string = "Which " + Style + " have you enjoyed recently?"
    Beer = st.selectbox(style_string, (data[data['Style'] == Style]['Beer Name (Full)'].unique()))

    user_input = Beer

    # Locate numerical features of user inputted beer
    test_data = data[data["Beer Name (Full)"] == user_input]
    num_input = df_num.loc[test_data.index].values
    
    # Detect beer style
    style_input = test_data['Style'].iloc[0]

    return num_input, style_input

num_input, style_input = user_input_features()

##########


# Generate recommendations based on user input
def get_neighbors(data, num_input, style_input, same_style=False):
    if same_style==True:
        # Locate beers of same style
        df_target = data[data["Style"] == style_input].reset_index(drop=True)
    else:
        # Locate beers of different styles
        df_target = data[data["Style"] != style_input].reset_index(drop=True)

    df_target_num = df_num.loc[df_target.index]
    # Calculate similarities
    search = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(df_target_num)
    _ , queried_indices = search.kneighbors(num_input)
    # Top 5 recommendations
    target_rec_df = df_target.loc[queried_indices[0][1:]]
    target_rec_df = target_rec_df.sort_values(by=['review_overall'], ascending=False)
    target_rec_df = target_rec_df[['Name', 'Brewery', 'Style', 'review_overall']]
    target_rec_df.index = range(1, 6)
    target_rec_df.drop('review_overall', axis=1, inplace=True)
    return target_rec_df


########## Main Panel
st.markdown("\n")
st.markdown("\n")

# Add button to generate recommendations
st.write("Ready to check out your recommendations?")
display_recommendation_now = st.button('Beer me!')
if display_recommendation_now:
    # Generate recommendations
    st.header('Recommended Beers:')

    # List recommended beers with the same style
    st.subheader('Fancy the same style?')
    top_5_same_style_rec = get_neighbors(data, num_input, style_input, same_style=True)
    top_5_same_style_rec


    # List recommended beers with different styles
    st.subheader('Looking for something different?')
    top_5_diff_style_rec = get_neighbors(data, num_input, style_input, same_style=False)
    top_5_diff_style_rec

##########


st.write('---')
st.caption("All code and data available in the [project Github repository](https://github.com/ruthgn/Beer-Recommender).")
