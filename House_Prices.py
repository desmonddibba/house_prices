{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ee4fcb98",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.ensemble import VotingClassifier\n",
    "\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21f9c39e",
   "metadata": {},
   "source": [
    "# Kunskapskontroll"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9db5c1e-04b4-4663-bd52-99c7cfbf3871",
   "metadata": {},
   "source": [
    "### Fyll i uppgifterna nedan innan du lämnar in på LearnPoint: \n",
    "Namn på samtliga gruppmedlemmar: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c9240d4-6646-48ae-8837-45a0f0b34827",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "affcc534-883c-4184-86f3-dab20f4348ab",
   "metadata": {},
   "source": [
    "# Project Steps\n",
    "\n",
    "* Import & Analyze the Data\n",
    "* Analyze features & measue feature importance\n",
    "* Clean up data missing values\n",
    "* Transform Data\n",
    "* Build pipeline (-> Transform data, Scale Data )\n",
    "* GridSearch parametergrid for models\n",
    "* Validate model\n",
    "* Test model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a96931ff",
   "metadata": {},
   "source": [
    "# Code"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2a79328-c117-4258-948d-bd3d23dc24e8",
   "metadata": {},
   "source": [
    "#### Loading Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "147ea5de",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>longitude</th>\n",
       "      <th>latitude</th>\n",
       "      <th>housing_median_age</th>\n",
       "      <th>total_rooms</th>\n",
       "      <th>total_bedrooms</th>\n",
       "      <th>population</th>\n",
       "      <th>households</th>\n",
       "      <th>median_income</th>\n",
       "      <th>median_house_value</th>\n",
       "      <th>ocean_proximity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-122.23</td>\n",
       "      <td>37.88</td>\n",
       "      <td>41.0</td>\n",
       "      <td>880.0</td>\n",
       "      <td>129.0</td>\n",
       "      <td>322.0</td>\n",
       "      <td>126.0</td>\n",
       "      <td>8.3252</td>\n",
       "      <td>452600.0</td>\n",
       "      <td>NEAR BAY</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-122.22</td>\n",
       "      <td>37.86</td>\n",
       "      <td>21.0</td>\n",
       "      <td>7099.0</td>\n",
       "      <td>1106.0</td>\n",
       "      <td>2401.0</td>\n",
       "      <td>1138.0</td>\n",
       "      <td>8.3014</td>\n",
       "      <td>358500.0</td>\n",
       "      <td>NEAR BAY</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-122.24</td>\n",
       "      <td>37.85</td>\n",
       "      <td>52.0</td>\n",
       "      <td>1467.0</td>\n",
       "      <td>190.0</td>\n",
       "      <td>496.0</td>\n",
       "      <td>177.0</td>\n",
       "      <td>7.2574</td>\n",
       "      <td>352100.0</td>\n",
       "      <td>NEAR BAY</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-122.25</td>\n",
       "      <td>37.85</td>\n",
       "      <td>52.0</td>\n",
       "      <td>1274.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>558.0</td>\n",
       "      <td>219.0</td>\n",
       "      <td>5.6431</td>\n",
       "      <td>341300.0</td>\n",
       "      <td>NEAR BAY</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-122.25</td>\n",
       "      <td>37.85</td>\n",
       "      <td>52.0</td>\n",
       "      <td>1627.0</td>\n",
       "      <td>280.0</td>\n",
       "      <td>565.0</td>\n",
       "      <td>259.0</td>\n",
       "      <td>3.8462</td>\n",
       "      <td>342200.0</td>\n",
       "      <td>NEAR BAY</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \\\n",
       "0    -122.23     37.88                41.0        880.0           129.0   \n",
       "1    -122.22     37.86                21.0       7099.0          1106.0   \n",
       "2    -122.24     37.85                52.0       1467.0           190.0   \n",
       "3    -122.25     37.85                52.0       1274.0           235.0   \n",
       "4    -122.25     37.85                52.0       1627.0           280.0   \n",
       "\n",
       "   population  households  median_income  median_house_value ocean_proximity  \n",
       "0       322.0       126.0         8.3252            452600.0        NEAR BAY  \n",
       "1      2401.0      1138.0         8.3014            358500.0        NEAR BAY  \n",
       "2       496.0       177.0         7.2574            352100.0        NEAR BAY  \n",
       "3       558.0       219.0         5.6431            341300.0        NEAR BAY  \n",
       "4       565.0       259.0         3.8462            342200.0        NEAR BAY  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Importing Data\n",
    "imported_data = pd.read_csv(r'C:\\Users\\Admin\\MLProjects\\House-Prices\\kunskapskontroll_ai2_del1\\housing.csv')\n",
    "imported_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "968fb620-8b2e-435d-a1ad-531020e9e611",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>longitude</th>\n",
       "      <th>latitude</th>\n",
       "      <th>housing_median_age</th>\n",
       "      <th>total_rooms</th>\n",
       "      <th>total_bedrooms</th>\n",
       "      <th>population</th>\n",
       "      <th>households</th>\n",
       "      <th>median_income</th>\n",
       "      <th>median_house_value</th>\n",
       "      <th>rooms_per_household</th>\n",
       "      <th>bedrooms_per_household</th>\n",
       "      <th>population_per_household</th>\n",
       "      <th>ocean_proximity_&lt;1H OCEAN</th>\n",
       "      <th>ocean_proximity_INLAND</th>\n",
       "      <th>ocean_proximity_ISLAND</th>\n",
       "      <th>ocean_proximity_NEAR BAY</th>\n",
       "      <th>ocean_proximity_NEAR OCEAN</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-122.23</td>\n",
       "      <td>37.88</td>\n",
       "      <td>41.0</td>\n",
       "      <td>880.0</td>\n",
       "      <td>129.0</td>\n",
       "      <td>322.0</td>\n",
       "      <td>126.0</td>\n",
       "      <td>8.3252</td>\n",
       "      <td>452600.0</td>\n",
       "      <td>6.984127</td>\n",
       "      <td>1.023810</td>\n",
       "      <td>2.555556</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-122.22</td>\n",
       "      <td>37.86</td>\n",
       "      <td>21.0</td>\n",
       "      <td>7099.0</td>\n",
       "      <td>1106.0</td>\n",
       "      <td>2401.0</td>\n",
       "      <td>1138.0</td>\n",
       "      <td>8.3014</td>\n",
       "      <td>358500.0</td>\n",
       "      <td>6.238137</td>\n",
       "      <td>0.971880</td>\n",
       "      <td>2.109842</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-122.24</td>\n",
       "      <td>37.85</td>\n",
       "      <td>52.0</td>\n",
       "      <td>1467.0</td>\n",
       "      <td>190.0</td>\n",
       "      <td>496.0</td>\n",
       "      <td>177.0</td>\n",
       "      <td>7.2574</td>\n",
       "      <td>352100.0</td>\n",
       "      <td>8.288136</td>\n",
       "      <td>1.073446</td>\n",
       "      <td>2.802260</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-122.25</td>\n",
       "      <td>37.85</td>\n",
       "      <td>52.0</td>\n",
       "      <td>1274.0</td>\n",
       "      <td>235.0</td>\n",
       "      <td>558.0</td>\n",
       "      <td>219.0</td>\n",
       "      <td>5.6431</td>\n",
       "      <td>341300.0</td>\n",
       "      <td>5.817352</td>\n",
       "      <td>1.073059</td>\n",
       "      <td>2.547945</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-122.25</td>\n",
       "      <td>37.85</td>\n",
       "      <td>52.0</td>\n",
       "      <td>1627.0</td>\n",
       "      <td>280.0</td>\n",
       "      <td>565.0</td>\n",
       "      <td>259.0</td>\n",
       "      <td>3.8462</td>\n",
       "      <td>342200.0</td>\n",
       "      <td>6.281853</td>\n",
       "      <td>1.081081</td>\n",
       "      <td>2.181467</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \\\n",
       "0    -122.23     37.88                41.0        880.0           129.0   \n",
       "1    -122.22     37.86                21.0       7099.0          1106.0   \n",
       "2    -122.24     37.85                52.0       1467.0           190.0   \n",
       "3    -122.25     37.85                52.0       1274.0           235.0   \n",
       "4    -122.25     37.85                52.0       1627.0           280.0   \n",
       "\n",
       "   population  households  median_income  median_house_value  \\\n",
       "0       322.0       126.0         8.3252            452600.0   \n",
       "1      2401.0      1138.0         8.3014            358500.0   \n",
       "2       496.0       177.0         7.2574            352100.0   \n",
       "3       558.0       219.0         5.6431            341300.0   \n",
       "4       565.0       259.0         3.8462            342200.0   \n",
       "\n",
       "   rooms_per_household  bedrooms_per_household  population_per_household  \\\n",
       "0             6.984127                1.023810                  2.555556   \n",
       "1             6.238137                0.971880                  2.109842   \n",
       "2             8.288136                1.073446                  2.802260   \n",
       "3             5.817352                1.073059                  2.547945   \n",
       "4             6.281853                1.081081                  2.181467   \n",
       "\n",
       "   ocean_proximity_<1H OCEAN  ocean_proximity_INLAND  ocean_proximity_ISLAND  \\\n",
       "0                      False                   False                   False   \n",
       "1                      False                   False                   False   \n",
       "2                      False                   False                   False   \n",
       "3                      False                   False                   False   \n",
       "4                      False                   False                   False   \n",
       "\n",
       "   ocean_proximity_NEAR BAY  ocean_proximity_NEAR OCEAN  \n",
       "0                      True                       False  \n",
       "1                      True                       False  \n",
       "2                      True                       False  \n",
       "3                      True                       False  \n",
       "4                      True                       False  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = imported_data\n",
    "\n",
    "# Creating new feature columns in the Dataset\n",
    "data['rooms_per_household'] = data['total_rooms'] / data['households']\n",
    "data['bedrooms_per_household'] = data['total_bedrooms'] / data['households']\n",
    "data['population_per_household'] = data['population'] / data['households']\n",
    "\n",
    "\n",
    "# Transforming 'ocean_proximity' feature using dummy variable encoding\n",
    "data = pd.get_dummies(data, columns=['ocean_proximity'])\n",
    "\n",
    "data.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "65f42da7-2f62-4fc1-982b-37b123f5ced9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ocean_proximity_INLAND           bool\n",
      "ocean_proximity_ISLAND           bool\n",
      "ocean_proximity_NEAR BAY         bool\n",
      "ocean_proximity_NEAR OCEAN       bool\n",
      "median_house_value            float64\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "# List of features used for the training data:\n",
    "\n",
    "housing_features = [\n",
    "    'ocean_proximity_INLAND', \n",
    "    'ocean_proximity_ISLAND', \n",
    "    'ocean_proximity_NEAR BAY', \n",
    "    'ocean_proximity_NEAR OCEAN',\n",
    "    'median_house_value'\n",
    "]\n",
    "housing = data[housing_features]\n",
    "print(housing.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2de1972f-3143-439f-a0a3-54516d535764",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Defining Median House Value as target variable.\n",
    "X = housing.drop('median_house_value', axis=1)\n",
    "y = housing['median_house_value']\n",
    "\n",
    "# X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e3c8ff3-3fd5-4a9c-a830-1dd9d1dc64d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# y.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "505bce40-a9f7-4dcd-a031-bab8d633925c",
   "metadata": {},
   "source": [
    "#### Train Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "1677719e-4c3f-435f-a4ba-3a27b6ec374b",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fee19169",
   "metadata": {},
   "source": [
    "## EDA"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eb25c7bb-ba01-4aeb-92ff-c798ff9bbbea",
   "metadata": {},
   "source": [
    "#### Checking Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5e566f5f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ocean_proximity_INLAND        0\n",
       "ocean_proximity_ISLAND        0\n",
       "ocean_proximity_NEAR BAY      0\n",
       "ocean_proximity_NEAR OCEAN    0\n",
       "median_house_value            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking for missing Values\n",
    "housing.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "64a2e9b3-5096-40e8-a3ca-9b00ab40b9c7",
   "metadata": {},
   "source": [
    "#### Inspecting Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a395cdb9-222f-4f7a-ab7b-7257ff1caa8f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Inspecting Dataset\n",
    "print(housing.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "38d6f450-20e8-4079-afd6-21290455212f",
   "metadata": {},
   "outputs": [],
   "source": [
    "sns.set(style=\"whitegrid\")\n",
    "correlation_matrix = housing.corr()\n",
    "target_correlations = correlation_matrix['median_house_value'].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f373da37-80de-4072-80f0-1d7d95e1f93b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Correlation Barplot\n",
    "bar_plot = sns.barplot(x=target_correlations.values, y=target_correlations.index, palette=\"coolwarm\")\n",
    "\n",
    "plt.title(\"Correlation of Features with House Price\")\n",
    "bar_plot.set_title(\"Correlation of Features with House Price\", fontsize=16)\n",
    "bar_plot.set_xlabel(\"Correlation\")\n",
    "bar_plot.set_ylabel(\"Features\")\n",
    "bar_plot.tick_params(labelsize=12)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68034eff-8284-4d8d-85e3-141c43c7c707",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "b1a3739d-bc1d-4546-996c-64d6a3484961",
   "metadata": {},
   "source": [
    "## Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e560d8fd-c2d9-45bb-9234-7cb48446fd57",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "d5438ce3-b866-45dc-990d-36bc6929299b",
   "metadata": {},
   "outputs": [],
   "source": [
    "svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "9e6958d8-5d58-4343-981f-a25916f52719",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearSVC(max_iter=100, random_state=42, tol=20)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearSVC</label><div class=\"sk-toggleable__content\"><pre>LinearSVC(max_iter=100, random_state=42, tol=20)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearSVC(max_iter=100, random_state=42, tol=20)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# estimators = [random_forest_clf, extra_trees_clf, svm_clf]\n",
    "\n",
    "# for estimator in estimators:\n",
    "   # print(\"Training the\", estimator)\n",
    "  #  estimator.fit(X_train, y_train)\n",
    "svm_clf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "bea42f56-b543-4395-969f-11bcbf34a54b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.011303996770286637"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# [estimator.score(X_val, y_val) for estimator in estimators]\n",
    "svm_clf.score(X_val, y_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "77b44f21-da9e-401b-85b1-e2c2456d8467",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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EREREVUwJuFEREREJDiRgK9PUVBQgLCwMHTt2hWWlpYYM2YM0tPTi53//PlzzJgxA7a2trC1tYWvry/evn37iVFIYxJORERERIIrgEiw16dYt24dIiMjsXjxYuzZswcqKioYP348cnNzZc6fMmUK7t27h23btiEsLAxnz55FQEDAJ8UgC5NwIiIiIhJcgYCvssrNzcWWLVvg7e2N7t27w9zcHKGhocjIyMDRo0el5l+6dAn/+9//EBISgtatW6NTp05YtGgR4uLikJGR8QmRSOONmURERESk1BwdHeVuP378uMzx69ev482bN+jYsaN4TEdHB61atcKFCxcwYMAAifnJyckwMDBA06ZNxWMdOnSAiooKUlJS4OTk9AlnIYlJOBEREREJThke1vP48WMAgLGxscS4oaEhHj16JDU/IyNDaq66ujrq1Kkjc/6nYBJORERERIITconC4irdH/Pu3TsAhYn0v2loaODly5cy5/93btH8nJycMsVQHPaEExEREVGVpKmpCQBSN2Hm5OSgRo0aMufLumEzJycHNWvWFDQ2JuFEREREJLgCFeFeZVXUWpKZmSkxnpmZCSMjI6n5RkZGUnNzc3Px4sUL1KtXr+yByMAkvILcv38fZmZmOH/+fJmPYWZmhujoaABAXl4etm3bJlB0wPnz52FmZoagoKCPvnfRuRT3Gjt2rNT+Z8+ehZmZGSZNmiTz+B4eHlLHadOmDRwcHBAUFIT3799/9ByuXLkCb29vdOrUCW3btkWfPn2wbNkyPHv2TOb848ePY+zYsejYsSOsra0xZMgQREZGoqDg/75AW716tdxzPXDggNRxZ8yYATMzMxw7dkxqW2muMxERUWWmDEsUmpubQ0tLSyL/evXqFf7880/Y2NhIzbe1tcXjx48l1hEv2rddu3ZljkMW9oRXEGNjYyQlJaF27dplPkZSUhK0tbUBAImJiQgJCcGoUaMEirDQzp070bdvX5l/MP9r9erVsLa2lhqX1UsVHR0NU1NTnDx5EhkZGTI/Tfbv3x8LFiwQ//z27VskJSUhJCQE+fn5WLhwYbGxxMbGYsGCBRg6dCg2bNgAPT093Lx5E+vXr0dCQgK2bNmC5s2bi+cvW7YM4eHh+O677zBr1ixoamri7NmzWLJkCf744w+JJNnIyAj79++X+b46OjoSP2dnZ+PYsWMwNTVFREQE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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df = X.copy()\n",
    "df['target'] = y\n",
    "\n",
    "correlation_matrix = df.corr()\n",
    "sns.heatmap(correlation_matrix, annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3df46894-4645-4bc7-866e-1ec605c884d7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d945a736-a0fc-44ac-abb9-7083081e30f0",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3336e4ab-2229-48da-a9cf-6330ad02ad5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# named_estimators = [\n",
    " #    (\"random_forest_clf\", random_forest_clf),\n",
    " #    (\"extra_trees_clf\", extra_trees_clf),\n",
    " #    (\"svm_clf\", svm_clf)\n",
    "# ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f2cacd2-74d9-4d5a-95fb-13bf7704c3b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "voting_clf = VotingClassifier(named_estimators)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48b5e50f-a539-400d-872d-891cc861a9b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "voting_clf.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58945b33-2a53-4c1b-a149-1124fe1df8a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "voting_clf.score(X_val, y_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "490b4f35-96c0-410f-b44e-43079a374704",
   "metadata": {},
   "outputs": [],
   "source": [
    "[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6066001f-527c-4ec4-8989-b0fd985a5e90",
   "metadata": {},
   "outputs": [],
   "source": [
    "extra_trees_clf.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54b2896a-173a-402e-b490-8fe005c4862c",
   "metadata": {},
   "source": [
    "## Stramlit APP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98d9646e-f541-4a68-a952-36012d47ee1c",
   "metadata": {},
   "outputs": [],
   "source": [
    " # Creating Streamlit App\n",
    "nav = st.sidebar.radio(\"Navigation Menu\",[\"Teoretiska frågor\", \"Data & Modelling\", \"Rapport\", \"Presentation\"])\n",
    "\n",
    "if nav == \"Teoretiska frågor\":\n",
    "    st.title(\"G7 besvarade teoretiska frågor:\")\n",
    "    st.write(\"\"\"•  Vad är en CSV fil?\n",
    "•\tEn CSV-fil (Comma Separated Values) är en särskild filtyp som lagrar information som är avgränsad av kommatecken istället för kolumner.\n",
    "•  Kalle delar upp sin data i ”Träning”, ”Validering” och ”Test”, vad används respektive del för?\n",
    "•\tTräning: Används för att träna modellerna på en del av datasetet.\n",
    "•\tValidering: Används för att jämföra modeller mot varandra och se vilken av modellerna som hanterar ny data bäst.\n",
    "•\tTest: Används för att slutligen utvärdera den modell som klarade valideringstestet bäst.\n",
    "•  Julia delar upp sin data i träning och test. På träningsdatan så tränar hon tre modeller; ”Linjär Regression”, ”Lasso regression” och en ”Random Forest modell”. Hur skall hon välja vilken av de tre modellerna hon skall fortsätta använda när hon inte skapat ett explicit ”validerings-dataset”?\n",
    "•\tHon kan utvärdera modellernas prestanda genom att använda ett accuracy score eller annan relevant metrik på testdatan för att välja den bästa modellen.\n",
    "•  Vad är ”regressionsproblem\"? Kan du ge några exempel på modeller som används och potentiella tillämpningsområden?\n",
    "•\tRegressionsproblem kännetecknas av att den beroende variabeln kan anta kontinuerliga värden påverkade av ett eller flera oberoende variabler.\n",
    "•\tExempel på modeller: Linjär regression, Lasso regression, Support Vector Regressor, Decision Tree Regressor.\n",
    "•\tTillämpningsområden: Fastighetsvärdering, försäljningsprognoser, temperaturförutsägelser.\n",
    "•  Hur kan du tolka RMSE och vad används det till?\n",
    "•\tRMSE (Root Mean Squared Error) är ett mått på hur stor avvikelsen är i genomsnitt mellan de observerade värdena och de förutsagda värdena i en regressionsmodell. Det används för att skatta de bästa parametrarna i modellen. Formeln är:\n",
    "\n",
    " \n",
    "\n",
    "•  Vad är ”klassificeringsproblem\"? Kan du ge några exempel på modeller som används och potentiella tillämpningsområden? Vad är en ”Confusion Matrix”?\n",
    "•\tKlassificeringsproblem innebär att den beroende variabeln kan anta två eller flera klasser eller kategorier.\n",
    "•\tExempel på modeller: Logistic Regression, Decision Tree Classifier, Support Vector Classifier (SVC), Voting Classifier.\n",
    "•\tTillämpningsområden: E-postspamfiltrering, sjukdomsdiagnos, kreditriskbedömning.\n",
    "•\tEn Confusion Matrix är ett sätt att visualisera modellens prestanda genom att jämföra de predikterade klasserna med de faktiska klasserna.\n",
    "•  Vad är Streamlit för något och vad kan det användas till?\n",
    "•\tStreamlit är ett open-source Python-ramverk för att skapa interaktiva applikationer som visualiserar och manipulerar data.\n",
    "•  Vad kännetecknar en nominell variabel?\n",
    "•\tEn nominell variabel är en kategorisk variabel som kan anta ett bestämt antal diskreta värden utan någon inbördes ordning.\n",
    "•  Vad kännetecknar en ordinal variabel?\n",
    "•\tEn ordinal variabel är en kategorisk variabel som kan rangordnas eller rankas (till exempel förstaplats, andraplats, tredjeplats).\n",
    "•  Förklara (gärna med ett exempel): Ordinal encoding, one-hot encoding, dummy variable encoding.\n",
    "•\tOrdinal encoding: Varje kategori får ett numeriskt värde. Exempel:\n",
    "o\tColors: red=1, blue=2, green=3.\n",
    "•\tOne-hot encoding: Varje kategori representeras som en binär kolumn. Exempel:\n",
    "o\t[['red'], ['green'], ['blue']]\n",
    "o\t[[0, 0, 1], [0, 1, 0], [1, 0, 0]]\n",
    "•\tDummy variable encoding: Liknar one-hot encoding men en kategori utelämnas för att undvika multikollinearitet. Exempel:\n",
    "o\t[['red'], ['green'], ['blue']]\n",
    "o\t[[0, 1], [1, 0], [0, 0]]\n",
    "•  Om vi använder vanlig linjär regression, skall vi använda one-hot-encoding eller dummy variable encoding?\n",
    "•\tVid användning av vanlig linjär regression är det rekommenderat att använda dummy variable encoding för att undvika multikollinearitet.\n",
    "\n",
    " \"\"\")\n",
    "    st.write(\"Desmond Dibba, Joakim Kvistholm, Johan Gentle-Hilton, Celeste Sun, Shiva Moradi\")\n",
    "   \n",
    "\n",
    "if nav == \"Data & Modelling\":\n",
    "    st.title(\"Data & Machine Learning Modelling\")\n",
    "    st.write('In this section we will look at the data, Analyse the data and also modell it by using Linear Regression, SVM, ...')\n",
    "             \n",
    "    st.header(\"Data\")\n",
    "    st.subheader(\"Scatterplot of the Data\")\n",
    "    st.pyplot(fig_data)\n",
    "    \n",
    "    st.subheader(\"Raw Data\")\n",
    "    st.write(\"If you want to see the raw data, check the box below.\")\n",
    "    if st.checkbox('Show raw data'):\n",
    "        st.write(modelling_data)\n",
    "        \n",
    "    st.header(\"Machine Learning Model - Linear Regression\")\n",
    "    st.subheader(\"Visualizing the Model\")\n",
    "    st.pyplot(fig_model)\n",
    "    \n",
    "    st.subheader(\"Linear Regression prediction Interface\")\n",
    "    val = st.number_input(\"Enter the x value you want to predict.\", step = 0.25)\n",
    "    val = np.array(val).reshape(1, -1)\n",
    "    prediction = lin_reg.predict(val)\n",
    "    if st.button(\"Predict\"):\n",
    "        st.success(f\"The predicted y value is: {prediction}\")\n",
    "\n",
    "    st.header(\"Machine Learning Model - SVM\")\n",
    "    st.subheader(\"Visualizing the Model\")\n",
    "    st.pyplot(fig_model)\n",
    "    \n",
    "    st.subheader(\"SVM prediction Interface\")\n",
    "    val = st.number_input(\"Enter the x value you want to predict.\", step = 0.25)\n",
    "    val = np.array(val).reshape(1, -1)\n",
    "    prediction = lin_reg.predict(val)\n",
    "    if st.button(\"Predict\"):\n",
    "        st.success(f\"The predicted y value is: {prediction}\")\n",
    "    \n",
    "if nav == \"Rapport\":\n",
    "    st.title(\"Läs vår grupps rapporten\")\n",
    "    st.write(\"\"\"[https://docs.streamlit.io/library/get-started](https://docs.streamlit.io/library/get-started) .\"\"\")\n",
    "    \n",
    "\n",
    "    \n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
