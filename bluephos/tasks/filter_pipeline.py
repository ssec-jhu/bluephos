import pandas as pd
from dplutils.pipeline import PipelineTask
from bluephos.modules.filter import FilterTask


# Helper function to create both filter_in and filter_out tasks
def filter(df: pd.DataFrame, column: str, threshold: float, filter_in=True) -> pd.DataFrame:
    filter_task = FilterTask(column_name=column, threshold=threshold, filter_in=filter_in)
    return filter_task.task(df)


# Dynamically create filter_in and filter_out tasks for NN
FilterNNInTask = PipelineTask(
    "filter_nn_in",
    filter,
    kwargs={"column": "z"},
    context_kwargs={"threshold": "t_nn"},
    # filter_in
)

FilterNNOutTask = PipelineTask(
    "filter_nn_out",
    filter,
    kwargs={"column": "z", "filter_in": False},
    context_kwargs={"threshold": "t_nn"},
    # filter_out
)

# Dynamically create filter_in and filter_out tasks for XTB
FilterXTBInTask = PipelineTask(
    "filter_xtb_in",
    filter,
    kwargs={"column": "ste"},
    context_kwargs={"threshold": "t_ste"},
    # filter_in
)

FilterXTBOutTask = PipelineTask(
    "filter_xtb_out",
    filter,
    kwargs={"column": "ste", "filter_in": False},
    context_kwargs={"threshold": "t_ste"},
    # filter_out
)

# Dynamically create filter_in and filter_out tasks for DFT
FilterDFTInTask = PipelineTask(
    "filter_dft_in",
    filter,
    kwargs={"column": "dft_energy_diff"},
    context_kwargs={"threshold": "t_dft"},
    # filter_in
)

FilterDFTOutTask = PipelineTask(
    "filter_dft_out",
    filter,
    kwargs={"column": "dft_energy_diff", "filter_in": False},
    context_kwargs={"threshold": "t_dft"},
    # filter_out
)
