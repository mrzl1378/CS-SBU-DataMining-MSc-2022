# Project 1 - Sheedeh Sharif Bakhtiar
# Student ID: 400422108

##############################

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, chisquare, spearmanr, kendalltau, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def get_dataframe(path="./AB_NYC_2019.csv"):
    """Return Pandas Dataframe from CSV file."""
    return pd.read_csv(path)


def remove_na_values(dataframe):
    """Remove NA values from Pandas Dataframe."""
    fixable_columns = ["last_review", "reviews_per_month"]
    for column in fixable_columns:
        dataframe[column] = dataframe[column].fillna(0)
    return dataframe.dropna()


def remove_outliers(dataframe):
    """Remove outliers from Pandas Dataframe."""
    z_score = "Z Score"
    dataframe[z_score] = 0
    neighbourhood_group = "neighbourhood_group"
    neighbourhood_groups = dataframe[
        neighbourhood_group
    ].unique()  # Get all unique neighbourhood groups
    for neighbourhood in neighbourhood_groups:
        z_score_array = stats.zscore(
            dataframe.loc[dataframe[neighbourhood_group] == neighbourhood]["price"]
        )  # Calculate Z-Score on a neighbourhood group-basis
        dataframe.loc[
            dataframe[neighbourhood_group] == neighbourhood, z_score
        ] = z_score_array  # Set Z-score
    return dataframe[np.abs(dataframe[z_score]) < 2]


def clean_data(dataframe):
    """Clean Pandas Dataframe data."""
    dataframe = remove_na_values(dataframe)
    dataframe = dataframe.drop(
        columns=["latitude", "longitude", "host_name", "name", "last_review"], axis=1
    )
    return remove_outliers(dataframe)


def create_neighbourhood_graphs(dataframe):
    """Create graphs from neighbourhood data and save to disk"""
    neighbourhood_group = "neighbourhood_group"
    neighbourhood_groups = dataframe[
        neighbourhood_group
    ].unique()  # Get all unique neighbourhood groups
    mean_prices = []

    # Get list of mean price per neighbourhood group
    for neighbourhood in neighbourhood_groups:
        mean_prices.append(
            dataframe.loc[dataframe[neighbourhood_group] == neighbourhood][
                "price"
            ].mean()
        )

    # Visualize mean price per neighbourhood group
    ax = sns.barplot(x=neighbourhood_groups, y=mean_prices)
    ax.set(xlabel="Neighbourhood Group", ylabel="Mean Price")
    ax.get_figure().savefig("mean_price_per_group.png")

    # Visualize number of listings
    n_listings = [
        dataframe.loc[dataframe[neighbourhood_group] == neighbourhood].shape[0]
        for neighbourhood in neighbourhood_groups
    ]
    ax = sns.barplot(x=neighbourhood_groups, y=n_listings)
    ax.set(xlabel="Neighbourhood Group", ylabel="Number of Listings")
    ax.get_figure().savefig("n_listing_per_group.png")

    # Visualize number of listings per single neighbourhood
    neighbourhoods = "neighbourhood"
    u_neighbourhoods = dataframe[
        neighbourhoods
    ].unique()  # Get all unique neighbourhoods

    n_listings = [
        dataframe.loc[dataframe[neighbourhoods] == neighbourhood].shape[0]
        for neighbourhood in u_neighbourhoods
    ]
    ax = sns.barplot(x=u_neighbourhoods, y=n_listings)
    ax.set(xlabel="Neighbourhoods", ylabel="Number of Listings", xticklabels=[])
    ax.get_figure().savefig("n_listing_per_neighbourhood.png")

    # Visualize room type frequency per neighbourhood group
    room_types = dataframe["room_type"].unique()
    room_type_aggr_data = [
        (
            room_type,
            pd.DataFrame(
                {
                    "Room Type Count": [
                        dataframe.loc[
                            (dataframe["neighbourhood_group"] == group)
                            & (dataframe["room_type"] == room_type)
                        ].shape[0]
                        for group in neighbourhood_groups
                    ],
                    "Neighbourhood Group": neighbourhood_groups,
                }
            ),
        )
        for room_type in room_types
    ]
    concat_room_types = pd.concat(
        [
            dataframe.assign(Dataset=room_type)
            for room_type, dataframe in room_type_aggr_data
        ]
    )
    concat_room_types.columns.values[-1] = "Room Type"
    grid = sns.FacetGrid(
        concat_room_types, col="Room Type", hue="Room Type", height=5.0, aspect=1.5
    )
    grid.map(sns.barplot, "Neighbourhood Group", "Room Type Count")
    grid.savefig("room_type_per_group.png")


def gaussian_distribution_test(dataframe, key="price", n_samples=5000):
    """Check if values of a single column fall under Gaussian distribution."""
    price_column = dataframe[key].sample(n=n_samples)
    _, p = normaltest(price_column, axis=None)
    if p > 0.05:
        print("{}: Probably Gaussian".format(key))
    else:
        print("{}: Probably not Gaussian".format(key))


def min_max_normalize(dataframe, column="price"):
    """Normalize values of a single column using min-max normalization."""
    return (dataframe[column] - dataframe[column].min()) / (
        dataframe[column].max() - dataframe[column].min()
    )


def chi_square_test(dataframe, key="price"):
    """Check if values of a single column fall under chi-squared distribution."""
    sample_prices = min_max_normalize(dataframe, key)
    _, p = chisquare(sample_prices)
    if p > 0.05:
        print("{}: Probably chi-square".format(key))
    else:
        print("{}: Probably not chi-square".format(key))


def spearman_correlation(dataframe, key1="number_of_reviews", key2="price"):
    """Check correlation between two columns using Spearman Coefficient."""
    stat, p = spearmanr(dataframe[key1], dataframe[key2])
    if p > 0.05:
        print("{} & {}: Probably independent".format(key1, key2))
    else:
        print("{} & {}: Probably dependent".format(key1, key2))


def kendall_correlation(dataframe, key1="number_of_reviews", key2="price"):
    """Check correlation between two columns using Kendall Coefficient."""
    stat, p = kendalltau(dataframe[key1], dataframe[key2])
    if p > 0.05:
        print("{} & {}: Probably independent".format(key1, key2))
    else:
        print("{} & {}: Probably dependent".format(key1, key2))


def nonparametric_mannwhitney(dataframe, key1="number_of_reviews", key2="price"):
    """Check if values of two columns have the same distribution."""
    stat, p = mannwhitneyu(dataframe[key1], dataframe[key2])
    if p > 0.05:
        print("{} & {}: Probably the same distribution".format(key1, key2))
    else:
        print("{} & {}:Probably different distributions".format(key1, key2))


def prepare_data_for_model(
    dataframe,
    columns_to_drop=["id", "host_id"],
    columns_to_onehot=["neighbourhood_group", "room_type", "neighbourhood"],
    train_test_split_percent=0.2,
):
    """Prepare data in dataframe to be used for model training.

    Parameters
    ----------
    dataframe: Pandas.DataFrame
        The dataframe to be prepared
    columns_to_drop: list
        Columns to remove from dataframe before splitting
    columns_to_onehot: list
        Columns containing qualitative values (such as strings) that should be replaced with one-hot replacements
    train_test_split_percent: float
        Percentile of data to be set aside for testing data
    """
    data = dataframe.drop(columns=columns_to_drop, axis=1)
    for column in columns_to_onehot:
        one_hot_groups = pd.get_dummies(dataframe[column])
        data = data.drop(column, axis=1)
        data = data.join(one_hot_groups)

    # Split features/labels
    X = data.drop("price", axis=1)
    y = data["price"]

    # Split train/test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_test_split_percent, random_state=0
    )
    return X_train, X_test, y_train, y_test


def linear_model(X_train, X_test, y_train, y_test):
    """Train linear regression model."""
    linear_model = LinearRegression()  # Create the model
    linear_model.fit(X_train, y_train)  # Fit the model to our data

    # Predict
    y_pred = linear_model.predict(X_test)

    # Evaluate
    print(
        "Linear Regression Mean Absolute Error:",
        metrics.mean_absolute_error(y_test, y_pred),
    )
    print(
        "Linear Regression Mean Squared Error:",
        metrics.mean_squared_error(y_test, y_pred),
    )
    print(
        "Linear Regression Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    )


def random_forest_model(X_train, X_test, y_train, y_test):
    """Train random forest regressor."""
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train, y_train)

    # Predict
    y_pred = random_forest.predict(X_test)

    # Evaluate
    print(
        "Random Forest Mean Absolute Error:",
        metrics.mean_absolute_error(y_test, y_pred),
    )
    print(
        "Random Forest Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred)
    )
    print(
        "Random Forest Root Mean Squared Error:",
        np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    )


#########################3

if __name__ == "__main__":
    ##### Question 1
    df = get_dataframe()
    df = clean_data(df)

    ##### Question 2
    create_neighbourhood_graphs(df)

    ##### Question 3, 4 -> In Jupyter notebook

    ##### Question 5
    gaussian_distribution_test(df)
    chi_square_test(df)
    spearman_correlation(df)
    kendall_correlation(df)
    nonparametric_mannwhitney(df)

    ##### Question 6
    X_train, X_test, y_train, y_test = prepare_data_for_model(df)
    linear_model(X_train, X_test, y_train, y_test)
    random_forest_model(X_train, X_test, y_train, y_test)
