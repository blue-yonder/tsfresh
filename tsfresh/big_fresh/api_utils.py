import base64
import json
import cloudpickle as pickle
import uuid

import pandas as pd
import zlib

TURN_ON_COMPRESSION=True


def get_dataframe_from_user(request):
    """
    Use the given request to extract the dataframe passed by the user.
    Later, the user should have the following possibilities:
      * direct string data in the json
      * s3 bucket
      * more ???
    In the moment, only the direct stream through the API gateway is implemented.
    :return the extracted dataframe
    """
    if request.stream:
        df = pd.read_csv(request.stream)
        return df
    else:
        return None


class DataHandle:
    def __init__(self, request=None):
        if request:
            self.data_handle = request.args["data_token"]
        else:
            self.data_handle = str(uuid.uuid4())
        self.chunk_number = None
        self.completed_chunks = []
        self.calculation_started = None

    def __str__(self):
        return json.dumps(dict(
            data_handle=self.data_handle,
            chunk_number=self.chunk_number,
            calculation_started=str(self.calculation_started),
        ))

    def is_finished(self):
        return True
        return self.chunk_number and len(self.completed_chunks) == self.chunk_number


def write_status(df, data_handle):
    pass


def load_status(data_handle):
    pass


def update_status(data_handle, **kwargs):
    pass


def encode_payload(payload):
    pickled_payload = pickle.dumps(payload)
    pickled_payload_as_bytes = base64.b64encode(pickled_payload)
    pickled_payload_as_string = pickled_payload_as_bytes.decode()

    if TURN_ON_COMPRESSION:
        compressed_string = zlib.compress(pickled_payload_as_string)
        return compressed_string
    else:
        return pickled_payload_as_string


def decode_payload(compressed_string):
    if TURN_ON_COMPRESSION:
        pickled_payload_as_string = zlib.decompress(compressed_string)
    else:
        pickled_payload_as_string = compressed_string
    pickled_payload_as_bytes = pickled_payload_as_string.encode()
    pickled_payload = base64.b64decode(pickled_payload_as_bytes)
    payload = pickle.loads(pickled_payload)
    return payload
