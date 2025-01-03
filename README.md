## Overview

The Personalized Recipe Recommendation System aims to suggest recipes tailored to the individual preferences of the user. It considers factors like ingredients on hand, dietary preferences, and past recipe ratings to recommend meals that match their tastes. The system integrates a machine learning model that learns from user interactions, providing increasingly accurate suggestions over time.

## Approach

The project followed a modular approach with multiple components working together for a seamless experience:

1. **Data Collection**: Used recipe1M+ dataset containing over a million recipes to train the recommendation model. The dataset included recipe details, ingredients, links, and receipe.
   
2. **Data Processing**: The raw data was cleaned and preprocessed to make it ready for machine learning. Ingredient lists were parsed, and each recipe was encoded using a set of features representing the ingredients and recipe attributes (e.g., cuisine, difficulty).

    ### Dietary Restriction Classification
- **Approach**: Recipes are classified into dietary categories (e.g., vegetarian, vegan) by checking the ingredients against predefined sets for each restriction.
- **Process**: For each recipe, ingredients are checked for categories like "non-vegetarian" or "eggitarian" first. If none match, the function checks for other categories (e.g., vegan, gluten-free). Dummy variables are created for each category and added to the dataset.

    ### Cuisine Classification
- **Approach**: Recipes are classified into cuisines (e.g., Italian, Indian) based on their ingredients.
- **Process**: Each recipe’s ingredients are compared to typical ingredients for each cuisine. Recipes with matching ingredients are assigned the corresponding cuisine(s). Dummy variables for each cuisine are added to the dataset.

    ### Recipe Complexity Classification
- **Approach**: Recipes are categorized as easy, medium, or hard based on the number of directions provided.
- **Process**: The number of directions in each recipe determines the complexity. Recipes with 3 or fewer directions are "easy", 4–7 are "medium", and more than 7 are "hard".

    ### Final Data Processing
- **Process**: A new dataset is created, combining relevant columns (e.g., title, NER, dietary restrictions, cuisines, complexity) for model building. Dummy columns for dietary restrictions and cuisines are included to enable machine learning models.

    ### TF-IDF Vectorization
- **Approach**: Ingredients are vectorized using the TF-IDF method to convert text data into numerical form for machine learning.

    ### One-Hot Encoding
- **Approach**: Categorical variables like dietary restrictions and cuisines are one-hot encoded to convert them into numerical form for machine learning.


3. **Modeling**: 
   - **Content-based Filtering**: A content-based approach was also incorporated, focusing on the recipe's ingredients and matching them to those preferred by the user in the past. The model uses cosine similarity to recommend recipes based on ingredient overlap. 



## Challenges

### 1. **Data Quality**
   - *Challenge*: 1. Quantity and quality of data were inconsistent, leading to incomplete or inaccurate recommendations. 
   2. Lack of Labelled Data: The dataset did not have explicit labels for dietary restrictions, cuisines, or recipe complexity, making it challenging to classify recipes accurately.
   - *Solution*: I partitioned into a sizable chunk so that we can preprocess a small sample first then scale it up to the entire dataset. I also labeled based on patterns and common ingredients to classify recipes into dietary restrictions and cuisines.

### 2. **Recommendation Accuracy**
   - *Challenge*: The model is struggeling to show accurate recommendations based on user preferences and past interactions.
   - *Solution*: With more time could build a robust machine learning model that learns from user interactions and feedback to provide more accurate recommendations. Also, I could use more advanced techniques like collaborative filtering and deep learning models like transformers to improve the recommendation accuracy.

### 3. **Computation Complexity**
   - *Challenge*: Processing a large dataset with millions of recipes and ingredients required significant computational resources and time.
   - *Solution*: I tried to parallelize the processing steps and optimized the code to reduce the time complexity of the algorithms used for classification and recommendation but eventually decided to work with a smaller subset of the data for faster prototyping.


## Ideas for Improvement

1. **Enhanced Personalization**: With more time, I would integrate more personalized filters based on factors like health conditions (e.g., gluten-free, low-carb) and seasonal preferences.

2. **Advanced Model Integration**: I would explore using deep learning techniques such as neural networks to predict recipes based on more complex patterns in user preferences and ingredient combinations.
Techinicques like collaborative filtering, matrix factorization, and deep learning could be explored for more accurate recommendations.
Also ensemble methods like bagging like random forest and boosting like gradient boosting could be used to improve the model performance.

3. **Real-time Feedback Loop**: Incorporating a real-time feedback loop, where the system continuously learns from the user’s evolving preferences, would improve the system’s accuracy.Implementing ci/cd pipeline to deploy the model in production environment.

4. **Deployment**: Deploying the model as a web application or would make it more accessible to users, allowing them to receive recipe recommendations on the go.


## How to run jupyter notebook
0. **Clone repository**: 
   Clone the repository to your local machine using the following command:
   ```bash
   git clone
   ```
   make sure you have git lfs installed to clone the repository.
   
1. **Create a virtual environment**: 
   Create a virtual environment and install the required packages some of them are listed:
   ```
   numpy, pandas, matplotlib, seaborn, scikit-learn, nltk, spacy, ast
   ```
   or you can also run requirements.txt file to install all the required packages.

2. **Create folders**: 
   File structure should be as follows:
   ```
    data
    ├── full_dataset.csv
    notebooks
    ├── recipe_recommendation.ipynb
    ├── nlp_review.ipynb
    ├── data_visulization.ipynb
    output
    ├── empty folder
    ```
3. **Run the notebook**: 
   Run the jupyter notebook `recipe_recommendation.ipynb` to see the code implementation and results.

4. **Show Visulization**:
    Run the jupyter notebook `data_visulization.ipynb` to see the data visulization.








