from src.programs.bps import *
from src.programs.fractions.gpt_utils import *
from src.programs.prior import *
from src.programs.teacher import (
    GPTProgramTeacher,
    ProbabilisticProgramTeacher,
    RandomProgramTeacher,
    RankingProgramTeacher,
)
from src.programs.utils import *


def initialize_teacher(strategy, dataset, populations, *args, **kwargs):
    if strategy == "random":
        teacher = RandomProgramTeacher(dataset, *args)
    elif strategy == "probabilistic":
        teacher = ProbabilisticProgramTeacher(dataset, populations, *args, **kwargs)
    elif strategy == "ranking":
        teacher = RankingProgramTeacher(dataset, populations, *args, **kwargs)
    elif strategy == "gpt":
        gpt_helper = FractionGPTHelper()
        teacher = GPTProgramTeacher(gpt_helper, dataset, *args, **kwargs)
    elif strategy == "gemma+bayesian":
        gpt_helper = FractionGPTHelper()
        teacher = GPTProgramTeacher(gpt_helper, dataset, *args, **kwargs)
    elif strategy == "gemma+oracle":
        gpt_helper = FractionGPTHelper()
        teacher = GPTProgramTeacher(gpt_helper, dataset, *args, **kwargs)
    elif strategy == "gemma":
        gpt_helper = FractionGPTHelper()
        teacher = GPTProgramTeacher(gpt_helper, dataset, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized strategy: {strategy}")
    return teacher
