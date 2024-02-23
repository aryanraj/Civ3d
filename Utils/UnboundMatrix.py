from typing import Dict, Tuple
from scipy import sparse as sp
from scipy.sparse._index import IndexMixin
import numpy as np
import numpy.typing as npt

__all__ = ["unbound_matrix"]


class UnboundIndexMixin(IndexMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index: Dict[Tuple[int, int], int] = {}

    def _validate_indices(self, key):
        """
        This method is overidden to avoid throwing error when the index is
        higher than self.shape. We are also not interested in negative indices.
        """
        from scipy.sparse._base import _spbase
        from scipy.sparse._index import (isintlike, _boolean_index_to_array,
                   _unpack_index, _compatible_boolean_index)
        if (isinstance(key, (_spbase, np.ndarray)) and
                key.ndim == 2 and key.dtype.kind == 'b'):
            if key.shape != self.shape:
                raise IndexError('boolean index shape does not match array shape')
            row, col = key.nonzero()
        else:
            row, col = _unpack_index(key)
        M, N = self.shape


        def _validate_bool_idx(
            idx: npt.NDArray[np.bool_],
            axis_size: int,
            axis_name: str
        ) -> npt.NDArray[np.int_]:
            if len(idx) != axis_size:
                raise IndexError(
                    f"boolean {axis_name} index has incorrect length: {len(idx)} "
                    f"instead of {axis_size}"
                )
            return _boolean_index_to_array(idx)


        if isintlike(row):
            row = int(row)
        elif (bool_row := _compatible_boolean_index(row)) is not None:
            row = _validate_bool_idx(bool_row, M, "row")
        elif not isinstance(row, slice):
            row = self._asindices(row, M)


        if isintlike(col):
            col = int(col)
        elif (bool_col := _compatible_boolean_index(col)) is not None:
            col = _validate_bool_idx(bool_col, N, "column")
        elif not isinstance(col, slice):
            col = self._asindices(col, N)

        return row, col

    def _get_intXint(self, row, col):
        i = self._index.get((row, col), -1)
        return 0 if i == -1 else self.data[i]


    def _get_intXarray(self, row, col):
        raise NotImplementedError()


    def _get_intXslice(self, row, col):
        raise NotImplementedError()


    def _get_sliceXint(self, row, col):
        raise NotImplementedError()


    def _get_sliceXslice(self, row, col):
        raise NotImplementedError()


    def _get_sliceXarray(self, row, col):
        raise NotImplementedError()


    def _get_arrayXint(self, row, col):
        raise NotImplementedError()


    def _get_arrayXslice(self, row, col):
        raise NotImplementedError()


    def _get_columnXarray(self, row, col):
        raise NotImplementedError()


    def _get_arrayXarray(self, row, col):
        raise NotImplementedError()


    def _set_intXint(self, row, col, x):
        i = self._index.get((row, col), -1)
        if i == -1:
            self.data = np.append(self.data, x)
            self.row = np.append(self.row, row)
            self.col = np.append(self.col, col)
            self._shape = (np.max(self.row) + 1, np.max(self.col) + 1)
            self._index[(row, col)] = self.data.shape[0] - 1
        else:
            self.data[i] = x


    def _set_arrayXarray(self, row, col, x):
        raise NotImplementedError()


class unbound_matrix(UnboundIndexMixin, sp.coo_matrix):
    """
    A sparse matrix in COOrdinate format with no upperbound in size

    The UnboundMatrix adds ability to update scipy.sparse.coo_matrix after the
    object initilization. We do this by updating the self.data, self.coords,
    and self._shape attributes directly.

    This may be dangerous as we are going to be missing the additional safety
    put while initialization. However, having tight control about the dtype, we
    can forgo the safety.

    Initially, it is an empty matrix of shape (1,1), with dtype
    """

    def __init__(self, dtype: npt.DTypeLike):
        super().__init__((1, 1), np.dtype(dtype))


if __name__ == "__main__":
    matrix = unbound_matrix(np.float64)
    matrix[0, 1] = 10
    assert np.allclose(matrix.todense(), [[0, 10]])

    matrix[0, 1] = 20
    assert np.allclose(matrix.todense(), [[0, 20]])

    matrix[[0, 1], 1] = 10
    assert matrix.todense() == [[0, 10], [0, 10]]

    matrix[np.ix_([0, 1], [0, 1])] = 20
    assert matrix.todense() == [[20, 20], [20, 20]]

    matrix[np.ix_([0, 1], [1])] += 20
    assert matrix.todense() == [[40, 20], [40, 20]]

