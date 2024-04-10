import pandas as pd
from emi1d.modeling import emi1d_forward, dataframe_modeling
from emi1d.inversion import emi1d_invert, dataframe_inversion


# Input arguments
df = pd.read_csv('emi1d_measurements.csv')
inversion_weights = {
    "Q_HCP_9000_2.0": 1.0,
    "Q_HCP_9000_4.0": 1.0,
    "Q_HCP_9000_8.0": 1.0,
    "Q_PRP_9000_2.0": 1.0,
    "Q_PRP_9000_4.0": 1.0,
    "Q_PRP_9000_8.0": 1.0,
}

inversion_settings = {
    "lambda": 10,
    "weights": inversion_weights,
}

# Initialize
model_instance = dataframe_modeling(df)  # check modeling_settings are correct 
#model_instance = emi1d_forward(df)  # check modeling_settings are correct 
# I could use emi1d_forward and make it possible to initialize the model_instance
# with a dataframe. Then, I don't have to create a modeling class
# initializing the model_instance with the metadata of the dataframe is very important
# in the inversion process. Model and inversion must match
inversion_instance = emi1d_invert(model_instance, **inversion_settings)

dataframe_inversion = dataframe_inversion(model_instance, inversion_instance)
df = dataframe_inversion.invert_dataframe(df)

df.to_csv('emi1d_inversion.csv')
