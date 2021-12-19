<h1 align="center">
    🍺 Beer-Recommender 🍺
</h1>

<p align="center">
  <a target="_blank" href="https://share.streamlit.io/ruthgn/beer-recommender/main/beer-recommender-app.py">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" width="170px;" alt="Launch Streamlit Web App" />
  </a>
</p>


## About The App
This is a beer recommender app that generates two sets of beer recommendations based on user input. The first set will list recommended beers of user's **chosen style**, while the second one will list recommended beers of **other styles** that are still ***within the user's chosen beer's taste profile***.

### Recommendation Engine
The recommendation engine powering the app is a content-based recommendation system that utilizes k-Nearest Neighbors algorithm to determine similar items. Items similar to user input are then ranked based on their overall consumer review scores to generate the final recommendations. To see how the recommendation system is built, check out this Kaggle [notebook](https://www.kaggle.com/ruthgn/creating-a-beer-recommender-deployment).

### Data
Recommendations are generated from a list of 3197 unique beers from 934 different breweries and take into consideration the following aspects of each beer:
* Alcohol content (% by volume)
* Minimum and maximum IBU (International Bitterness Units)
* Mouthfeel
  - Astringency
  - Body
  - Alcohol
* Taste
  - Bitter
  - Sweet
  - Sour
  - Salty
* Flavor And Aroma
  - Fruity
  - Hoppy
  - Spices
  - Malty

The source data is a slightly modified version of [this](https://www.kaggle.com/ruthgn/beer-profile-and-ratings-data-set) data set on Kaggle. A good number of beers in the original data set have only been reviewed by fewer than 25 people--making their overall favorability scores somewhat questionable, especially when compared to those displayed by beers that have been reviewed by a lot more users. To tackle this issue, the updated data set (used here) contains ML-generated overall scores for beers with fewer than 25 reviewers. To learn about the process in more detail, visit [this](https://www.kaggle.com/ruthgn/beer-score-prediction-nn-embedding-kerastuner) Kaggle notebook.

## Tech Stack
* Python
* NumPy
* pandas
* scikit-learn
* Streamlit


## Running The App Locally
Open a command prompt in your chosen project directory. Create a virtual environment with conda, then activate it.
```
conda create -n myenv python=3.9.7
conda activate myenv
```

Clone this git repo, then install the specified requirements with pip.
```
git clone https://github.com/ruthgn/Beer-Recommender
cd Beer-Recommender
pip install -r requirements.txt
```

Run the app.
```
streamlit run beer-recommender-app.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/ruthgn/Beer-Recommender/blob/main/LICENSE) file for details.
