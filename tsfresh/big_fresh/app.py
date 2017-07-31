from flask import request
import pandas as pd

from tsfresh.big_fresh.api_utils import DataHandle, write_status, load_status
from tsfresh.big_fresh.df_utils import extract_features_in_background, combine_files
from tsfresh.examples import load_robot_execution_failures


def add_views(app):
    @app.route('/', methods=['POST', 'PUT', 'GET'])
    def home():
        """
        Main method: calculate the features of the given csv dataframe.
        """
        data_handle = DataHandle()

        if not data_handle:
            # TODO: Better error message
            return "Malformed input", 500

        # TODO
        df, _ = load_robot_execution_failures()
        df = pd.melt(df, id_vars=["id", "time"])

        # Store the new data handle in the DB
        write_status(df, data_handle)

        # Start the feature calculation and hand back the handle to retrieve the data later
        extract_features_in_background(df, data_handle)

        # Tell the user about the calculation
        return str(data_handle)

    @app.route('/get_data', methods=['GET'])
    def get_data():
        """
        Check if the data is already available and if yes, return it.
        """
        data_handle = DataHandle(request)

        # Retrieve the newest DB changes of this handle
        load_status(data_handle)

        if data_handle.is_finished():
            # Get the data from S3 and return the combined data to the user
            return combine_files(data_handle)
        else:
            return "Not finished", 500



