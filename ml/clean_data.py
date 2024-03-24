import pandas as pd
import numpy as np
import logging
import sys

if __package__ is None or __package__ == '':
    from model import LOG_FILE
else:
    from .model import LOG_FILE

# Define logging handlers
logFileHandler = logging.FileHandler(LOG_FILE)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logFileHandler,
        consoleHandler
    ]
)

def basic_cleaning(df, output_path, target, test=False):
    '''
    Basic cleaning of data
    '''
    logging.info("[ START: Cleaning data ]")

    # The data may contain spaces
    # Strip leading and trailing spaces from all columns
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Remove spaces from column names and also replace hyphen with underscore
    # df.columns = df.columns.str.replace(" ", "")
    df = df.rename(columns={col_name: col_name.replace(' ', '') for col_name in df.columns})
    df = df.rename(columns={col_name: col_name.replace('-', '_') for col_name in df.columns})

    # Replace missing data with ?
    # df = df.replace('?', pd.NA)
    # Note: Replacing question marks with pd.NA casued problems when using OneHotEncoder
    
    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Filter categorical and numerical columns:
    catColumns = df.select_dtypes(include="object").columns.tolist()
    numColumns = df.select_dtypes(exclude="object").columns.tolist()
    logging.info(f"[ CONTROL: Categorical features: {catColumns}")
    logging.info(f"[ CONTROL: Numerical features: {numColumns}")

    # Return results when testing
    if test==False:
        try:
            df.to_csv(output_path, index=False)
            logging.info(f"[ Cleaned data saved to {output_path} ]")
            return df, catColumns, numColumns
        except:
            logging.error(f"[ Unable to save data to {output_path} ]")
    else:
        logging.info(f"[ Cleaned data returned ]")
        return df, catColumns, numColumns
