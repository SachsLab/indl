from typing import Optional, Tuple, List, Dict


__all__ = ['from_neuropype_h5']


def _recurse_get_dict_from_group(grp):
    import numpy as np
    result = dict(grp.attrs)
    for k, v in result.items():
        if isinstance(v, np.ndarray) and v.dtype.char == 'S':
            result[k] = v.astype('U').astype('O')
    for k, v in grp.items():
        result[k] = _recurse_get_dict_from_group(v)
    return result


def from_neuropype_h5(filename: str, chunk_names: List[str] = []) -> List[Tuple[str, dict]]:
    """
    Import a Neuropype-exported HDF5 file.

    Args:
        filename: Name of file on disk. Opened with h5py.File.
        chunk_names: Limit return to a subset of the chunks in the data file.

    Returns:
        A list of (name, chunk_dict) tuples.
    """
    import numpy as np
    import h5py
    from pandas import DataFrame
    f = h5py.File(filename, 'r')

    chunks = []
    if 'chunks' in f.keys():
        chunks_group = f['chunks']
        ch_keys = [_ for _ in chunks_group.keys() if _ in chunk_names]
        for ch_key in ch_keys:
            chunk_group = chunks_group.get(ch_key)

            # Process data
            block_group = chunk_group.get('block')
            data_ = block_group.get('data')
            if isinstance(data_, h5py.Dataset):
                data = data_[()]
            else:
                # Data is a group. This only happens with sparse matrices.
                import scipy.sparse
                data = scipy.sparse.csr_matrix((data_['data'][:], data_['indices'][:], data_['indptr'][:]),
                                               data_.attrs['shape'])

            axes_group = block_group.get('axes')
            axes = []
            for ax_ix, axis_key in enumerate(axes_group.keys()):
                axis_group = axes_group.get(axis_key)
                ax_type = axis_group.attrs.get('type')
                new_ax = {'name': axis_key, 'type': ax_type}
                if ax_type == 'axis':
                    new_ax.update(dict(x=np.arange(data.shape[ax_ix])))
                elif ax_type == 'time':
                    nom_rate = axis_group.attrs.get('nominal_rate')
                    if np.isnan(nom_rate):
                        nom_rate = None
                    new_ax.update(dict(nominal_rate=nom_rate,
                                       times=axis_group.get('times')[()]))
                elif ax_type == 'frequency':
                    new_ax.update(dict(frequencies=axis_group.get('frequencies')[()]))
                elif ax_type == 'space':
                    new_ax.update(dict(names=axis_group.get('names')[()],
                                       naming_system=axis_group.attrs['naming_system'],
                                       positions=axis_group.get('positions')[()],
                                       coordinate_system=axis_group.attrs['coordinate_system'],
                                       units=axis_group.get('units')[()]))
                elif ax_type == 'feature':
                    new_ax.update(dict(names=axis_group.get('names')[()],
                                       units=axis_group.get('units')[()],
                                       properties=axis_group.get('properties')[()],
                                       error_distrib=axis_group.get('error_distrib')[()],
                                       sampling_distrib=axis_group.get('sampling_distrib')[()]))
                elif ax_type == 'instance':
                    new_ax.update({'times': axis_group.get('times')[()]})
                    if 'instance_type' in axis_group.attrs:
                        new_ax.update({'instance_type': axis_group.attrs['instance_type']})
                    _dat = axis_group.get('data')[()]
                    if not _dat.dtype.names:
                        new_ax.update({'data': axis_group.get('data')[()]})
                    else:
                        _df = DataFrame(_dat)
                        # Convert binary objects to string objects
                        str_df = _df.select_dtypes([np.object])
                        str_df = str_df.stack().str.decode('utf-8').unstack()
                        for col in str_df:
                            _df[col] = str_df[col]
                        new_ax.update({'data': _df})

                elif ax_type == 'statistic':
                    new_ax.update(dict(param_types=axis_group.get('param_types')[()]))
                elif ax_type == 'lag':
                    new_ax.update(dict(xlags=axis_group.get('lags')[()]))
                if new_ax is not None:
                    axes.append(new_ax)

            chunks.append((ch_key, dict(data=data, axes=axes,
                                        props=_recurse_get_dict_from_group(chunk_group.get('props')))))

    return chunks
