import pandas as pd
from dplutils.pipeline import PipelineTask
from bluephos.modules.filter import FilterTask


def apply_filter(df: pd.DataFrame, column_name, threshold: float, filter_in: bool) -> pd.DataFrame:
        filter_task = FilterTask(column_name=column_name, threshold=threshold, filter_in=filter_in)
        return filter_task.task(df)

# Helper function to create both filter_in and filter_out tasks
def create_filter_task(task_name: str, column_name: str, threshold_key: str, filter_in: bool) -> PipelineTask:
    def apply_filter(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        filter_task = FilterTask(column_name=column_name, threshold=threshold, filter_in=filter_in)
        return df.apply(filter_task.task, axis=1)

    return PipelineTask(
        task_name,
        apply_filter,
        context_kwargs={
            "threshold": threshold_key,      # Get threshold from the context
        }
    )

# Dynamically create filter_in and filter_out tasks for NN
FilterNNInTask = create_filter_task(
    "apply_filter_nn_in", 
    "z", 
    "t_nn", 
    True  # filter_in
)

FilterNNOutTask = create_filter_task(
    "apply_filter_nn_out", 
    "z", 
    "t_nn", 
    False  # filter_out
)

# Dynamically create filter_in and filter_out tasks for XTB
FilterXTBInTask = create_filter_task(
    "apply_filter_xtb_in", 
    "ste", 
    "t_ste", 
    True  # filter_in
)

FilterXTBOutTask = create_filter_task(
    "apply_filter_xtb_out", 
    "ste", 
    "t_ste", 
    False  # filter_out
)

# Dynamically create filter_in and filter_out tasks for DFT
FilterDFTInTask = create_filter_task(
    "apply_filter_dft_in", 
    "dft_energy_diff", 
    "t_dft", 
    True  # filter_in
)

FilterDFTOutTask = create_filter_task(
    "apply_filter_dft_out", 
    "dft_energy_diff", 
    "t_dft", 
    False  # filter_out
)
