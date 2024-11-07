"""
This module contains utility functions and classes for data analysis and visualization.

It provides a `ProjectDataFrame` class that represents a project-specific DataFrame and includes methods for data manipulation, feature extraction, and visualization.

The `ProjectDataFrame` class allows users to initialize an instance with a DataFrame or a file path, set and get the underlying data, set the features and target variables, and perform various data analysis and visualization tasks.

The module also includes helper functions for classifying columns as categorical or numerical, generating bar plots, violin plots, chi-square contingency tables, and t-test results for specified features.

Many of these helper functions are written with general purpose in mind, so they can be directly used in other contexts as well.

"""

import itertools
from collections import namedtuple
import textwrap

import numpy as np
import pandas as pd

from IPython.display import display, HTML, Markdown
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


pd.options.display.float_format = "{:.4f}".format


class ProjectDataFrame:
    """
    A class representing a project-specific DataFrame.

    Attributes:
        df (pd.DataFrame): The DataFrame object containing the data.
        features (pd.DataFrame): The DataFrame object containing the features.
        target (pd.Series): The Series object containing the target variable.
        features_names (list): A list of feature names.
        target_name (str): The name of the target variable.
        random_state (int): The random state value used for initialization.

    Methods:
        __init__(self, df_or_path, random_state=42): Initializes the ProjectDataFrame object.
        get_data(self): Returns a copy of the DataFrame object.
        set_data(self, df): Sets the DataFrame object.
        get_features_target(self, target_name="diabetes"): Sets the features and target variables.
        categoricals_inspection(self, *, target_name="diabetes"): Performs inspection on categorical features.
        numericals_inspection(self, *, target_name="diabetes"): Performs inspection on numerical features.
        get_features_barplots(self, *, features_names, target_name="diabetes"): Generates barplots for specified features.
        get_feature_barplot(self, ax, *, feature_name, target_name="diabetes", annotate_p=False): Generates a barplot for a single feature.
        get_features_violinplots(self, *, features_names, target_name="diabetes"): Generates violin plots for specified features.
        get_feature_violinplot(self, ax, *, feature_name, target_name="diabetes"): Generates a violin plot for a single feature.
        get_features_chi2s_tables(self, *, features_names, target_name="diabetes"): Generates chi-square contingency tables for specified features.
        get_feature_chi2(self, *, feature_name, target_name="diabetes"): Performs chi-square test for a single feature.
        get_features_ttest_table(self, *, feature_names, target_name="diabetes"): Generates t-test results for specified features.
        get_feature_ttest(self, *, feature_name, target_name="diabetes"): Performs t-test for a single feature.
    """

    def __init__(self, df_or_path, random_state=42):
        """
        Initializes an instance of the class.

        Parameters:
        - df_or_path: Either a DataFrame or a file path to read the data from.
        - random_state: An integer representing the random state for reproducibility.

        Raises:
        - ValueError: If the input is neither a DataFrame nor a file path.
        """

        if isinstance(df_or_path, str):
            self.df = pd.read_csv(
                df_or_path, na_values=["No Info", "Other"], keep_default_na=False
            )
        elif isinstance(df_or_path, pd.DataFrame):
            self.df = df_or_path
        else:
            raise ValueError(
                "Invalid input. Please provide a DataFrame or a file path."
            )
        self.features = None
        self.target = None
        self.features_names = None
        self.target_name = None
        self.get_features_target()
        self.random_state = random_state

    # setter and getter
    def get_data(self):
        """
        Returns a copy of the underlying data.

        This method is a getter that returns a copy of the DataFrame object stored in the `df` attribute.
        It is designed to avoid unintentional modification of the underlying data by returning a copy instead of the original object.

        Returns:
            pandas.DataFrame: A copy of the underlying data.

        """
        return self.df.copy()

    def set_data(self, df):
        self.df = df
        return self.df

    # initialization helper method
    # explicitly set target_name to allow for more general purpose
    def get_features_target(self, target_name="diabetes"):
        """
        set the self.features and self.target from the dataframe.

        Parameters:
        target_name (str): The name of the target column. Default is "diabetes".

        Returns:
        None
        """

        if target_name not in self.df.columns:
            print(f"Target column {target_name} not found.")
            # no exception raised, because initialization calls with default target_name
            return
        self.features = self.df.drop(columns=[target_name])
        self.target = self.df[target_name]
        self.features_names = self.features.columns
        self.target_name = target_name
        return

    # presentation methods
    def dataset_overview(self):
        """
        Highest Level method, for the presentation purpose
        Generate an overview of the dataset.

        Returns:
            pandas.DataFrame: A DataFrame containing basic statistics and information about the dataset.
        """
        df = self.get_data()
        df.columns = [_title_case_conversion(i) for i in df.columns]
        # get the missing value count
        missing_values = df.isnull().sum()
        # get the unique value count
        unique_values = df.nunique()
        # get the count
        count = df.count()
        # get the mean, skip non numerical columns
        mean = df.select_dtypes(include=["number"]).mean()
        # get the std, skip non numerical columns
        std = df.select_dtypes(include=["number"]).std()
        # get the min, skip non numerical columns
        min_val = df.select_dtypes(include=["number"]).min()
        # get the max, skip non numerical columns
        max_val = df.select_dtypes(include=["number"]).max()
        # concatenate all the information
        overview = pd.concat(
            [
                missing_values,
                unique_values,
                count,
                mean,
                std,
                min_val,
                max_val,
            ],
            axis=1,
        )
        overview.columns = [
            "Missing Values",
            "Unique Values",
            "Count",
            "Mean",
            "Std",
            "Min",
            "Max",
        ]
        return overview

    def categoricals_inspection(self, *, target_name="diabetes"):
        """
        Highest Level method, for the presentation purpose
        Perform inspection and analysis on categorical features.

        Args:
            target_name (str, optional): The name of the target variable. Defaults to "diabetes".

        Returns:
            None
        """
        categoricals = _classify_columns(self.features)[0]
        self.get_features_chi2s_tables(
            features_names=categoricals, target_name=target_name
        )
        self.get_features_barplots(features_names=categoricals, target_name=target_name)
        return

    def numericals_inspection(self, *, target_name="diabetes"):
        """
        Highest Level method, for the presentation purpose
        Perform inspection on numerical features.

        Args:
            target_name (str, optional): Name of the target variable. Defaults to "diabetes".
        """

        numericals = _classify_columns(self.features)[1]
        self.get_features_violinplots(
            features_names=numericals, target_name=target_name
        )
        self.get_features_pairwise_kdeplots(
            features_names=numericals, target_name=target_name
        )
        self.get_features_pairwise_anova_tables(
            features_names=numericals, target_name=target_name
        )
        return

    # single feature test and visualization
    def get_features_barplots(self, *, features_names, target_name="diabetes"):
        """
        Generate bar plots for each feature in the given list of feature names,
        showing the prevalence of the target variable across categorical factors.

        Parameters:
        - features_names (list): A list of feature names for which to generate bar plots.
        - target_name (str): The name of the target variable. Default is "diabetes".

        Returns:
        None
        """
        max_units_per_row = 9  # Total units that can fit in one row
        current_units = 0
        rows = []
        current_row = []

        # Determine the layout for each feature
        for feature in features_names:
            unique_values = self.df[feature].nunique()
            required_units = unique_values + 1  # unique values + 1 for the plot

            if current_units + required_units > max_units_per_row:
                rows.append(current_row)
                current_row = []
                current_units = 0

            current_row.append((feature, required_units))
            current_units += required_units

        if current_row:
            rows.append(current_row)
        # debug
        # print(rows)

        fig = plt.figure(constrained_layout=True, figsize=(15, 5 * len(rows)))
        fig.suptitle(
            f"{_title_case_conversion(target_name)} Prevalence Across Categorical Factors",
            size="xx-large",
            weight="demi",
            # put at top middle
            x=0.5,
            y=1.05,
        )
        gs = fig.add_gridspec(len(rows), max_units_per_row)

        for row_index, row in enumerate(rows):
            col_index = 0
            for feature, units in row:
                ax = fig.add_subplot(gs[row_index, col_index : col_index + units])
                self.get_feature_barplot(
                    ax,
                    feature_name=feature,
                    target_name=target_name,
                    annotate_p=True,
                )
                col_index += units
        return

    def get_feature_barplot(
        self,
        ax,
        *,
        feature_name,
        target_name="diabetes",
        annotate_p=False,
    ):
        """
        Generate a bar plot to visualize the relationship between a feature and the target variable.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes on which to draw the plot.
        - feature_name (str): The name of the feature column in the DataFrame.
        - target_name (str, optional): The name of the target variable column in the DataFrame. Default is "diabetes".
        - annotate_p (bool, optional): Whether to annotate the p-values on the plot. Default is False.
        """

        ylim = 0.51  # hardcoding for now
        # in order to keep the order, we have to drop nan values manually
        unique_values = list(self.df[feature_name].dropna().unique())
        # order from the least frequent diabetes to most
        unique_values.sort(
            key=lambda x: self.df[self.df[feature_name] == x][target_name].mean()
        )
        sns.barplot(
            data=self.df,
            x=feature_name,
            y=target_name,
            hue=feature_name,
            palette="pastel",
            estimator="mean",
            dodge=False,
            ax=ax,
            order=unique_values,
        )
        _plot_params(
            ax,
            title=_title_case_conversion(feature_name),
            legend=False,
            ylabel=f"Proportion of {_title_case_conversion(target_name)}",
            ylim=ylim,
            # another temp hardcoding
            yticks=np.arange(0, ylim, 0.1),
            grid=True,
        )
        ax.title.set_text(_title_case_conversion(feature_name))
        if annotate_p:
            combinations = list(itertools.combinations(unique_values, 2))
            # hard coding for now
            anno_height = 0.14
            for pair in combinations:
                filtered_df = self.df[self.df[feature_name].isin(pair)]
                freq = pd.crosstab(filtered_df[feature_name], filtered_df[target_name])
                _, pval, _, _ = stats.chi2_contingency(freq)
                v1, v2 = pair
                x_axis_pair = unique_values.index(v1), unique_values.index(v2)
                # decide the height of the annotation automatically
                # hard coding for now
                _annotate_pval(ax, pval, x_axis_pair, anno_height)
                anno_height += 0.038

            pval = self.get_feature_chi2(
                feature_name=feature_name, target_name=target_name
            )[2]
        return

    def get_features_violinplots(self, *, features_names, target_name="diabetes"):
        """
        Generate violin plots to visualize the prevalence of the target variable across multiple features.

        Parameters:
        features_names (list): A list of feature names.
        target_name (str): The name of the target variable. Default is "diabetes".

        Returns:
        None
        """

        max_units_per_row = 4
        num_features = len(features_names)
        row_count = num_features // max_units_per_row
        fig, axes = plt.subplots(
            row_count, max_units_per_row, figsize=(15, 5 * row_count)
        )
        fig.suptitle(
            f"{_title_case_conversion(target_name)} Prevalence Across Features",
            size="xx-large",
            weight="demi",
            x=0.5,
            y=1.05,
        )
        axes_flat = axes.flatten()

        for idx, ax in enumerate(axes_flat):
            if idx >= num_features:
                break
            self.get_feature_violinplot(
                ax,
                feature_name=features_names[idx],
                target_name=target_name,
            )
        # Customize the legend
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            ["Negative", "Positive"],
            title=f"{_title_case_conversion(target_name)}",
            loc="lower center",
            ncol=2,
        )

        plt.show()
        return

    def get_feature_violinplot(self, ax, *, feature_name, target_name="diabetes"):
        """
        Generate a violin plot to visualize the distribution of a feature with respect to a target variable.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes on which to draw the violin plot.
        - feature_name (str): The name of the feature to plot.
        - target_name (str, optional): The name of the target variable. Default is "diabetes".

        Returns:
        None
        """

        sns.violinplot(
            data=self.df,
            x=np.ones(len(self.df)),
            y=feature_name,
            hue=target_name,
            palette="pastel",
            split=True,
            inner="quartile",
            ax=ax,
        )
        _plot_params(
            ax,
            title=_title_case_conversion(feature_name),
            legend=False,
            ylabel=_title_case_conversion(feature_name),
            grid=True,
        )
        ax.set_ylim(bottom=0, top=self.df[feature_name].max() * 1.1)
        return

    def get_features_chi2s_tables(self, *, features_names, target_name="diabetes"):
        """
        Generate contingency tables and test results for each feature in features_names.

        Parameters:
        - features_names (list): A list of feature names for which to generate the contingency tables and test results.
        - target_name (str): The name of the target variable. Default is "diabetes".

        Returns:
        None
        """
        for feature_name in features_names:
            freq, chi2, p_val, dof = self.get_feature_chi2(
                feature_name=feature_name, target_name=target_name
            )
            html_content = f"""
                        <h4>Gender vs diabetes diagnosis contingency table and test results:</h4>
                        {freq.to_html()}
                        <p>&chi;<sup>2</sup>: {chi2}<br>
                        <strong>p-value</strong>: {p_val}<br>
                        <strong>degrees of freedom</strong>: {dof}</p>
                        """
            display(HTML(html_content))
        return

    def get_feature_chi2(self, *, feature_name, target_name="diabetes"):
        """
        Calculate the chi-square test statistic and p-value for a given feature.

        Parameters:
        - feature_name (str): The name of the feature column in the DataFrame.
        - target_name (str): The name of the target column in the DataFrame. Default is "diabetes".

        Returns:
        - freq (pd.DataFrame): The contingency table of the feature and target columns.
        - chi2 (float): The chi-square test statistic.
        - p (float): The p-value of the chi-square test.
        - dof (int): The degrees of freedom of the chi-square test.
        """
        freq = pd.crosstab(self.df[feature_name], self.df[target_name])
        chi2, p, dof, _ = stats.chi2_contingency(freq)
        return (freq, chi2, p, dof)

    def get_features_ttest_table(self, *, feature_names, target_name="diabetes"):
        """
        Calculate the t-test statistics and p-values for a list of feature names.

        Parameters:
        - feature_names (list): A list of feature names.
        - target_name (str): The name of the target variable. Default is "diabetes".

        Returns:
        - ttests_results (DataFrame): A DataFrame containing the t-test statistics and p-values for each feature.
        """
        ttests_results = pd.DataFrame(
            data=np.nan, columns=["t_statistic", "p_value"], index=feature_names
        )
        for feature_name in feature_names:
            t_statistic, p_value = self.get_feature_ttest(
                feature_name=feature_name, target_name=target_name
            )
            ttests_results.loc[feature_name] = [t_statistic, p_value]
        return ttests_results

    def get_feature_ttest(self, *, feature_name, target_name="diabetes"):
        """
        Calculate the t-test statistic and p-value for a given feature.

        Parameters:
        feature_name (str): The name of the feature to calculate the t-test for.
        target_name (str, optional): The name of the target variable. Default is "diabetes".

        Returns:
        t_statistic (float): The t-test statistic.
        p_value (float): The p-value.
        """
        group1 = self.df[self.df[target_name] == 0].loc[:, feature_name]
        group2 = self.df[self.df[target_name] == 1].loc[:, feature_name]
        t_statistic, p_value = stats.ttest_ind(group1, group2)
        return t_statistic, p_value

    # pairwise feature test and visualization
    def get_features_pairwise_kdeplots(self, *, features_names, target_name="diabetes"):
        """
        Generate pairwise KDE plots for a given list of features.

        Parameters:
        -features_names (list): A list of feature names.
        -target_name (str): The name of the target variable. Default is "diabetes".

        Returns:
        None
        """

        max_units_per_row = 3
        # get the pair wise combination's count and tuple
        combinations = list(itertools.combinations(features_names, 2))
        num_coms = len(combinations)
        row_count = num_coms // max_units_per_row
        fig, axes = plt.subplots(
            row_count, max_units_per_row, figsize=(15, 5 * row_count)
        )
        axes_flat = axes.flatten()

        for idx, ax in enumerate(axes_flat):
            if idx >= num_coms:
                break
            n1, n2 = combinations[idx]
            self.get_feature_pairwise_kdeplot(
                ax,
                feature_name1=n1,
                feature_name2=n2,
                target_name=target_name,
            )
        fig.suptitle(
            "Cluster Visualization of features",
            size="xx-large",
            weight="demi",
            x=0.5,
            y=0.95,
        )
        plt.show()
        return

    def get_feature_pairwise_kdeplot(
        self, ax, *, feature_name1, feature_name2, target_name="diabetes"
    ):
        """
        Generate a pairwise KDE plot for two features.

        Parameters:
        - ax (matplotlib.axes.Axes): The axes on which to draw the plot.
        - feature_name1 (str): The name of the first feature.
        - feature_name2 (str): The name of the second feature.
        - target_name (str, optional): The name of the target variable. Defaults to "diabetes".

        Returns:
        None
        """

        # Legend for kdeplot can be tricky to deal with
        # so we edited df here directly to account for label generations
        df = self.get_data()
        # change binary 0 and 1 to negative and positive for df["target_name"]
        df[target_name] = df[target_name].replace({0: "Negative", 1: "Positive"})
        sns.kdeplot(
            ax=ax,
            data=df.sample(n=1000, random_state=self.random_state),
            x=feature_name1,
            y=feature_name2,
            hue=target_name,
            legend=True,
        )
        ax.set_title(
            f"{_title_case_conversion(feature_name1)} vs {_title_case_conversion(feature_name2)}"
        )
        ax.set_xlabel(_title_case_conversion(feature_name1))
        ax.set_ylabel(_title_case_conversion(feature_name2))
        return

    def get_features_pairwise_anova_tables(
        self, *, features_names, target_name="diabetes"
    ):
        """
        Generates a pairwise ANOVA table for a given list of feature names.

        Args:
            features_names (list): A list of feature names to compare pairwise.
            target_name (str, optional): The name of the target variable. Defaults to "diabetes".

        Returns:
            None
        """
        combinations = list(itertools.combinations(features_names, 2))
        for f1, f2 in combinations:
            anova_results = self.get_feature_pairwise_anova_table(
                feature_name1=f1, feature_name2=f2, target_name=target_name
            )
            table_title_html = f"<h2>{_title_case_conversion(f1)} versus {_title_case_conversion(f2)}</h2>"
            display(HTML(table_title_html))
            display(HTML(anova_results.to_html()))
        return

    def get_feature_pairwise_anova_table(
        self, *, feature_name1, feature_name2, target_name="diabetes"
    ):
        """
        Calculate the pairwise ANOVA for two features with respect to a target variable.

        Parameters:
        feature_name1 (str): The name of the first feature.
        feature_name2 (str): The name of the second feature.
        target_name (str, optional): The name of the target variable. Defaults to "diabetes".

        Returns:
        DataFrame: The pairwise ANOVA results.

        """
        model = ols(
            f"{target_name} ~ {feature_name1} * {feature_name2}",
            data=self.df.sample(10000, random_state=self.random_state),
        ).fit()
        return anova_lm(model, type=2)

    # normalization and preparation for PCA
    def normalize(self):
        """
        Normalize the data after handling missing values, encoding categorical columns, and scaling numerical columns.

        Returns:
            ProjNormalizedDF: A new instance of ProjNormalizedDF with the normalized data.

        Raises:
            ValueError: If the data contains missing values.
        """

        # check if the data has missing values
        if self.df.isnull().sum().any():
            raise ValueError("Data contains missing values, please handle them first.")
        categorical_cols, numerical_cols = _classify_columns(self.df)
        df_categoricals = categoricals_encoder(self.df, categorical_cols).reset_index(
            drop=True
        )
        df_numericals = numericals_scaler(self.df, numerical_cols).reset_index(
            drop=True
        )
        df = pd.concat([df_categoricals, df_numericals], axis=1)
        return ProjNormalizedDF(df)


class ProjNormalizedDF(ProjectDataFrame):
    """
    A class representing a normalized project DataFrame.

    This class inherits from the ProjectDataFrame class and provides additional method for performing PCA
    """

    def __init__(self, df):
        super().__init__(df)

    def perform_pca(self):
        """
        Perform Principal Component Analysis (PCA) on the features and target variables.

        Returns:
            PCA: The PCA object.
        """
        pca = PCA(self.features, self.target)
        return pca

    def normalize(self):
        """
        Raises:
            ValueError: If the data are already principal components, no normalization is needed.
        """
        # override the method to prevent normalization of already normalized data
        raise ValueError("Data are principal components. No normalization needed.")


class ProjPrincipalComponentsDF(ProjectDataFrame):
    """
    A class representing a project DataFrame with principal components as features
    """

    def __init__(self, df):
        super().__init__(df)

    def normalize(self):
        # override the method to prevent normalization of already normalized data
        raise ValueError("Data is already normalized.")


class PCA:
    """
    Principal Component Analysis (PCA) class for dimensionality reduction and visualization.

    Parameters:
    - features (numpy.ndarray or pandas.DataFrame): Input features for PCA.
    - target (numpy.ndarray or pandas.Series): Target variable for PCA.

    Attributes:
    - features (numpy.ndarray or pandas.DataFrame): Input features for PCA.
    - target (numpy.ndarray or pandas.Series): Target variable for PCA.
    - svd (namedtuple): Singular Value Decomposition (SVD) result containing U, S, and Vt matrices.
    - principal_components (numpy.ndarray): Principal components obtained from SVD.
    - explained_variance (numpy.ndarray): Explained variance ratio of each principal component.
    - X_pca_np (numpy.ndarray): Projected data onto the principal components.
    - X_pca_df (pandas.DataFrame): Projected data onto the principal components as a DataFrame.

    Methods:
    - get_loading_scores(): Get the loading scores of each feature on the principal components.
    - get_cumu_var_plot(): Plot the cumulative explained variance by principal components.
    - get_firt_two_components_scatter_plot(): Plot a scatter plot of the first two principal components.
    - get_project_principal_components_dataframe(): Get the projected data onto the principal components as a DataFrame.
    """

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.svd = self._perform_svd()
        self.principal_components = self.svd.Vt.T
        self.explained_variance = self.svd.S / np.sum(self.svd.S)
        self.X_pca_np = np.dot(self.features, self.principal_components)
        self.X_pca_df = pd.DataFrame(
            self.X_pca_np,
            columns=[
                f"No_{i+1} Pricinpal Component"
                for i in range(self.principal_components.shape[1])
            ],
        )

    def _perform_svd(self):
        """
        Perform Singular Value Decomposition (SVD) on the covariance matrix of the input features.

        Returns:
        - SVDResult: Namedtuple containing U, S, and Vt matrices obtained from SVD.
        """
        cov_matrix = np.cov(self.features, rowvar=False)
        SVDResult = namedtuple("SVDResult", ["U", "S", "Vt"])
        return SVDResult(*np.linalg.svd(cov_matrix))

    def get_loading_scores(self):
        """
        Get the loading scores of each feature on the principal components.

        Returns:
        - pandas.DataFrame: Loading scores of each feature on the principal components.
        """
        loading_scores_df = pd.DataFrame(
            self.principal_components,
            index=self.features.columns,
            columns=[f"PC{i+1}" for i in range(self.principal_components.shape[1])],
        )
        return loading_scores_df

    def get_cumu_var_plot(self):
        """
        Generates a plot of the cumulative explained variance by principal components.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.explained_variance), marker="o")
        plt.xlabel("Principal Components")
        plt.ylabel("Explained Variance")
        plt.title("Explained Variance by Principal Components")
        plt.grid(True)
        plt.show()
        return

    def get_firt_two_components_scatter_plot(self):
        """
        Generates a scatter plot of the first two principal components.
        """
        plt.figure(figsize=(10, 5))
        plt.scatter(
            self.X_pca_np[:, 0],
            self.X_pca_np[:, 1],
            c=self.target,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.title("PCA - First Two Principal Components")
        plt.colorbar(label="Diabetes Status")
        plt.grid(True)
        plt.show()
        return

    def get_project_principal_components_dataframe(self):
        """
        Get the projected data onto the principal components as a DataFrame.

        Returns:
        - ProjPrincipalComponentsDF: DataFrame containing the projected data onto the principal components.
        """
        df = pd.concat([self.X_pca_df, self.target], axis=1)
        return ProjPrincipalComponentsDF(df)


class LogisticModel:
    """
    A class representing a logistic regression model.

    Parameters:
    - features: The features used for training the model.
    - target: The target variable used for training the model.

    Attributes:
    - features: The features used for training the model.
    - target: The target variable used for training the model.
    - X_train: The training set features.
    - X_test: The test set features.
    - y_train: The training set target variable.
    - y_test: The test set target variable.
    """

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def logistic_training(
        self,
        random_state,
        test_size=0.2,
    ):
        """
        Train a logistic regression model.

        Parameters:
        - random_state (int): The seed used by the random number generator.
        - test_size (float): The proportion of the dataset to include in the test split.

        Returns:
        - model (LogisticRegression): The trained logistic regression model.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state
        )
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        return model

    def get_model_expression(self, model):
        """
        Get the logistic regression equation in LaTeX format.

        Parameters:
        - model: The trained logistic regression model.

        Returns:
        - Markdown: The logistic regression equation in LaTeX format.
        """
        # Extract coefficients and intercept
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        feature_names = (_title_case_conversion(c, "_") for c in self.features.columns)
        # Construct the logistic regression equation in LaTeX
        equation_logit = f"\\text{{logit}}(p) = {intercept:.3f}\\\\"
        for coef, name in zip(coefficients, feature_names):
            # decide if "+" or "-" should be used
            sign = "+" if coef > 0 else ""
            equation_logit += f" {sign} {coef:.3f} \\cdot \\text{{{name}}}\\\\"
        equation_prob = "p = \\frac{1}{1 + e^\\text{logit(p)}}"

        # Display the equations
        latex_content = textwrap.dedent(
            f"""
            ### Logistic Regression Equation (logit):
            $
            {equation_logit}
            $

            ### Logistic Regression Equation (probability):
            $
            {equation_prob}
            $
            """
        )

        return Markdown(latex_content)

    def evaluate(
        self,
        model,
        *,
        prob_threshhold=0.1,
        target_value_names=("Negative Diagnosis", "Positive Diagnosis"),
    ):
        """
        Predict the target values using the trained model.

        Parameters:
        - model: The trained logistic regression model.
        - prob_threshhold (float): The probability threshold for classification.
        - target_value_names (tuple): The names of the target values.

        Returns:
        - DataFrame: A classification report for the model.
        """
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        y_pred_threshold = (y_pred_proba >= prob_threshhold).astype(int)
        class_report = classification_report(
            self.y_test,
            y_pred_threshold,
            output_dict=True,
            target_names=target_value_names,
        )
        return pd.DataFrame(class_report).T


def categoricals_encoder(df, CATEGORICALS):
    """
    Encode categorical columns in a DataFrame using LabelEncoder.

    Parameters:
    - df: The DataFrame containing the categorical columns.
    - CATEGORICALS: A list of column names that are categorical.

    Returns:
    - df_categoricals: The DataFrame with encoded categorical columns.
    """
    label_encoder = LabelEncoder()
    df_categoricals = pd.DataFrame(columns=CATEGORICALS, dtype="int32")
    for cat in CATEGORICALS:
        df_categoricals.loc[:, cat] = label_encoder.fit_transform(df.loc[:, cat])
    return df_categoricals


def numericals_scaler(df, NUMERICALS):
    """
    Scale numerical columns in a DataFrame using StandardScaler.

    Parameters:
    - df: The DataFrame containing the numerical columns.
    - NUMERICALS: A list of column names that are numerical.

    Returns:
    - df_numericals: The DataFrame with scaled numerical columns.
    """
    scaler = StandardScaler()
    npa_numericals = scaler.fit_transform(df.loc[:, NUMERICALS])
    df_numericals = pd.DataFrame(npa_numericals, columns=NUMERICALS)
    return df_numericals


def hclustering(features):
    """
    Perform hierarchical clustering on a set of features and plot the dendrogram.

    Parameters:
    - features: The features to perform clustering on.

    Returns:
    - Z: The linkage matrix.
    """
    Z = linkage(features, method="ward")
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.show()
    return Z


def one_hot_encoder(df, column_name):
    """
    Perform one-hot encoding on a categorical column in a DataFrame.

    Parameters:
    - df: The DataFrame containing the categorical column.
    - column_name: The name of the categorical column.

    Returns:
    - df: The DataFrame with the one-hot encoded column.
    """
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(df[[column_name]])
    feature_names = encoder.get_feature_names_out([column_name])
    encoded_df = pd.DataFrame(encoded.A, columns=feature_names)
    df = df.drop(columns=[column_name]).reset_index(drop=True)
    # default to RangeIndex, which is better for concatenation
    df = pd.concat([df, encoded_df], axis=1)
    return df


def _classify_columns(df):
    """
    Classify columns in a DataFrame as categorical or numerical.

    Parameters:
    - df: The DataFrame to classify columns in.

    Returns:
    - categorical_cols: A list of column names classified as categorical.
    - numerical_cols: A list of column names classified as numerical.
    """
    # Initialize
    categorical_cols = []
    numerical_cols = []

    # Iterate through columns
    for col in df.columns:
        if df[col].dtype == "object":
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_values = df[col].nunique()
            if unique_values < 5:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    return categorical_cols, numerical_cols


def _plot_params(
    ax,
    title=None,
    legend=None,
    xlabel=None,
    xticks=None,
    ylabel=None,
    yticks=None,
    ylim=None,
    grid=None,
):
    """
    Helper function to set common plot aesthetic parameters.

    Parameters:
    - ax: The subplot to be modified.
    - title: The title of the subplot.
    - legend: Whether to include a legend in the subplot.
    - xlabel: The x-axis label.
    - xticks: The x-axis tick labels.
    - ylabel: The y-axis label.
    - yticks: The y-axis tick labels.
    - ylim: The y-axis limit.
    - grid: Whether to add a grid to the subplot.
    """
    # title = "Subplot title"
    if title is not None:
        ax.set_title(title, fontsize="x-large", weight="roman")

    # legend = False if legend should be removed
    if not legend:
        ax.legend_.remove()

    # xlabel = "x-axis title"
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize="large", weight="roman")
    else:
        ax.set_xlabel(None)

    # xticks = iterable of x-axis labels
    if xticks is not None:
        ax.set_xticks(ax.get_xticks(), xticks)

    # ylabel = "y-axis label"
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize="large", weight="roman")
    else:
        ax.set_ylabel(None)

    # yticks = iterable of numbers where y-ticks should go
    if yticks is not None:
        ax.set_yticks(yticks)

    # ylim = number
    if ylim is not None:
        ax.set_ylim(top=ylim)

    # grid = True if grid should be added to back of plot
    if grid is not None:
        ax.set_axisbelow(True)
        ax.grid(axis="y")
    return


def _annotate_pval(ax, pval, x_pair, y):
    """
    Annotate p-value ranges for statistical significance calculation and annotation.

    Parameters:
    - ax: The plot Axes object.
    - pval: The p-value or manual p-value.
    - x_pair: The (x1, x2) tuple of x values corresponding to variable pairs being compared.
    - y: The y value for annotation height.
    """
    # Gets level of statistical significance in star notation
    if 5.00e-02 < pval <= 1.00e00:
        star_annot = "ns"
    elif 1.00e-02 < pval <= 5.00e-02:
        star_annot = "*"
    elif 1.00e-03 < pval <= 1.00e-02:
        star_annot = "**"
    elif 1.00e-04 < pval <= 1.00e-03:
        star_annot = "***"
    elif pval <= 1.00e-04:
        star_annot = "****"

    # Line annotation showing pairs compared for statistical significance
    ax.plot(x_pair, (y, y), color="k", marker=3, linewidth=1)

    # Annotates statistical significance (star notation and p-value) on the passed axis
    if star_annot != "ns":
        y -= 0.005

    ax.annotate(
        f"{pval:.3f}\n{star_annot}",
        xy=((x_pair[0] + x_pair[1]) / 2, y + 0.005),
        color="k",
        size=10,
        weight="demi",
        horizontalalignment="center",
    )


def _title_case_conversion(s, delimiter="_", sep=" "):
    """
    Convert a string to title case.

    Parameters:
    - s: The string to convert.
    - delimiter: The delimiter used in the string.
    - sep: The separator to use in the converted string.

    Returns:
    - The converted string.
    """
    return sep.join([word.capitalize() for word in s.split(delimiter)])
