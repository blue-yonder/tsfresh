from glob import glob

import itertools

import pandas as pd

from tsfresh.big_fresh.api_utils import encode_payload, decode_payload, update_status
from tsfresh.feature_extraction.extraction import _do_extraction, _do_extraction_on_chunk
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfresh.utilities.distribution import Distributor, function_with_partly_reduce

from zappa.async import run


class ZappaDistributor(Distributor):
    def __init__(self, data_handle):
        Distributor.__init__(self, disable_progressbar=True, n_workers=0, progressbar_title="")

        # TODO: Only for testing
        from distributed import Client, LocalCluster
        self.client = Client(LocalCluster())

        self.data_handle = data_handle

    def distribute(self, func, partitioned_chunks):
        # We are cheating a bit here, as we do not feed the function func to the
        # clients, but our own version of it.
        # This is fine, as we actually know which function we are expecting
        calcs = []
        for chunk_list_id, chunk_list in enumerate(partitioned_chunks):
            kwargs = {"chunk_list_id": chunk_list_id, "chunk_list": chunk_list, "data_handle": self.data_handle}
            # TODO: Only for testing
            print(chunk_list_id, len(chunk_list))
            calcs.append(self.client.submit(extract_on_payload, **kwargs))

        self.client.gather(calcs)

        return []

    @staticmethod
    def encode_function(data):
        return encode_payload(data)

    @staticmethod
    def decode_function(payload):
        return decode_payload(payload)


def write_out_result(result, data_handle, chunk_list_id):
    encoded_result = encode_payload(result)

    filename = "tmp-output/{}-{}".format(data_handle.data_handle, chunk_list_id)
    with open(filename, "wb") as f:
        f.write(encoded_result)


def extract_on_payload(chunk_list_id, chunk_list, data_handle):
    kwargs = dict(default_fc_parameters=MinimalFCParameters(), kind_to_fc_parameters={})
    result = function_with_partly_reduce(chunk_list, map_function=_do_extraction_on_chunk,
                                         decode_function=decode_payload, kwargs=kwargs)

    update_status(data_handle=data_handle, completed_chunks=[chunk_list_id])
    write_out_result(result, data_handle=data_handle, chunk_list_id=chunk_list_id)


def extract_features_in_background(df, data_handle):
    distributor = ZappaDistributor(data_handle)
    _do_extraction(df, distributor=distributor,
                   column_id="id", column_value="value", column_kind="variable",
                   default_fc_parameters=MinimalFCParameters(), kind_to_fc_parameters={},
                   n_jobs=0, chunk_size=50, disable_progressbar=True)


def combine_files(data_handle):
    def get_contents():
        for file_name in glob("tmp-output/{}-*".format(data_handle.data_handle)):
            with open(file_name) as f:
                yield decode_payload(f.read())

    result = list(itertools.chain.from_iterable(get_contents()))
    result = pd.DataFrame(result)

    if len(result) != 0:
        result = result.pivot("id", "variable", "value")

    return result.to_csv()