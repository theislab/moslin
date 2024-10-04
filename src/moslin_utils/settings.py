# This file is a modified version of the original file from the Scanpy package.
from typing import Any


def _type_check(var: Any, varname: str, types: type | tuple[type, ...]):
    if isinstance(var, types):
        return
    if isinstance(types, type):
        possible_types_str = types.__name__
    else:
        type_names = [t.__name__ for t in types]
        possible_types_str = "{} or {}".format(", ".join(type_names[:-1]), type_names[-1])
    raise TypeError(f"{varname} must be of type {possible_types_str}")


class MoslinConfig:
    """Config manager for moslin."""

    def __init__(
        self,
        *,
        save_figures: bool = True,
    ):
        self.save_figures = save_figures

    @property
    def save_figures(self) -> bool:
        """Save figures to file."""
        return self._save_figures

    @save_figures.setter
    def save_figures(self, save_figures: bool):
        _type_check(save_figures, "save_figures", bool)
        self._save_figures = save_figures


settings = MoslinConfig()
