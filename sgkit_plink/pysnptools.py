"""PLINK 1.9 reader implementation"""
from pathlib import Path
from typing import Union

import dask.array as da
import dask.dataframe as dd
import numpy as np
from pysnptools.snpreader import Bed

from sgkit import create_genotype_call_dataset
from sgkit.api import DIM_SAMPLE

PathType = Union[str, Path]

FAM_FIELDS = [
    ("family_id", str, "U"),
    ("member_id", str, "U"),
    ("paternal_id", str, "U"),
    ("maternal_id", str, "U"),
    ("sex", str, "int8"),
    ("phenotype", str, "int8"),
]
FAM_DF_DTYPE = dict([(f[0], f[1]) for f in FAM_FIELDS])
FAM_ARRAY_DTYPE = dict([(f[0], f[2]) for f in FAM_FIELDS])

BIM_FIELDS = [
    ("contig", str, "U"),
    ("variant_id", str, "U"),
    ("cm_pos", "float32", "float32"),
    ("pos", "int32", "int32"),
    ("a1", str, "U"),
    ("a2", str, "U"),
]
BIM_DF_DTYPE = dict([(f[0], f[1]) for f in BIM_FIELDS])
BIM_ARRAY_DTYPE = dict([(f[0], f[2]) for f in BIM_FIELDS])


class BedReader(object):
    def __init__(self, path, shape, dtype=np.int8, count_A1=True):
        # n variants (sid = SNP id), n samples (iid = Individual id)
        n_sid, n_iid = shape
        # Initialize Bed with empty arrays for axis data, otherwise it will
        # load the bim/map/fam files entirely into memory (it does not do out-of-core for those)
        self.bed = Bed(
            str(path),
            count_A1=count_A1,
            # Array (n_sample, 2) w/ FID and IID
            iid=np.empty((n_iid, 2), dtype="str"),
            # SNP id array (n_variants)
            sid=np.empty((n_sid,), dtype="str"),
            # Contig and positions array (n_variants, 3)
            pos=np.empty((n_sid, 3), dtype="int"),
        )
        self.shape = (n_sid, n_iid, 2)
        self.dtype = dtype
        self.ndim = 3

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            raise IndexError(  # pragma: no cover
                f"Indexer must be tuple (received {type(idx)})"
            )
        if len(idx) != self.ndim:
            raise IndexError(  # pragma: no cover
                f"Indexer must be two-item tuple (received {len(idx)} slices)"
            )
        # Slice using reversal of first two slices --
        # pysnptools uses sample x variant orientation
        arr = self.bed[idx[1::-1]].read(dtype=np.float32, view_ok=False).val.T
        # Convert missing calls as nan to -1
        arr = np.nan_to_num(arr, nan=-1.0)
        arr = arr.astype(self.dtype)
        # Add a ploidy dimension, so allele counts of 0, 1, 2 correspond to 00, 01, 11
        arr = np.stack(
            [
                np.where(arr < 0, -1, np.where(arr == 0, 0, 1)),
                np.where(arr < 0, -1, np.where(arr == 2, 1, 0)),
            ],
            axis=-1,
        )

        # Apply final slice to 3D result
        return arr[:, :, idx[-1]]

    def close(self):
        # This is not actually crucial since a Bed instance with no
        # in-memory bim/map/fam data is essentially just a file pointer
        # but this will still be problematic if the an array is created
        # from the same PLINK dataset many times
        self.bed._close_bed()  # pragma: no cover


def _to_dict(df, dtype=None):
    return {
        c: df[c].to_dask_array(lengths=True).astype(dtype[c] if dtype else df[c].dtype)
        for c in df
    }


def read_fam(path: PathType, sep: str = "\t"):
    # See: https://www.cog-genomics.org/plink/1.9/formats#fam
    names = [f[0] for f in FAM_FIELDS]
    df = dd.read_csv(str(path) + ".fam", sep=sep, names=names, dtype=FAM_DF_DTYPE)

    def coerce_code(v, codes):
        # Set non-ints and unexpected codes to missing (-1)
        v = dd.to_numeric(v, errors="coerce")
        v = v.where(v.isin(codes), np.nan)
        return v.fillna(-1).astype("int8")

    df["paternal_id"] = df["paternal_id"].where(df["paternal_id"] != "0", None)
    df["maternal_id"] = df["maternal_id"].where(df["maternal_id"] != "0", None)
    df["sex"] = coerce_code(df["sex"], [1, 2])
    df["phenotype"] = coerce_code(df["phenotype"], [1, 2])

    return df


def read_bim(path: PathType, sep: str = " "):
    # See: https://www.cog-genomics.org/plink/1.9/formats#bim
    names = [f[0] for f in BIM_FIELDS]
    df = dd.read_csv(str(path) + ".bim", sep=sep, names=names, dtype=BIM_DF_DTYPE)
    df["contig"] = df["contig"].where(df["contig"] != "0", None)
    return df


def read_plink(
    path: PathType,
    chunks: Union[str, int, tuple] = "auto",
    fam_sep: str = "\t",
    bim_sep: str = " ",
    bim_int_contig: bool = False,
    count_a1: bool = True,
    lock: bool = False,
    persist: bool = True,
):
    # Load axis data first to determine dimension sizes
    df_fam = read_fam(path, sep=fam_sep)
    df_bim = read_bim(path, sep=bim_sep)

    # TODO: not on nearly 10x faster on real datasets
    if persist:
        df_fam = df_fam.persist()
        df_bim = df_bim.persist()

    arr_fam = _to_dict(df_fam, dtype=FAM_ARRAY_DTYPE)
    arr_bim = _to_dict(df_bim, dtype=BIM_ARRAY_DTYPE)

    # Load genotyping data
    call_genotype = da.from_array(
        # Make sure to use asarray=False in order for masked arrays to propagate
        BedReader(path, (len(df_bim), len(df_fam)), count_A1=count_a1),
        chunks=chunks,
        # Lock must be true with multiprocessing dask scheduler
        # to not get pysnptools errors (it works w/ threading backend though)
        lock=lock,
        asarray=False,
        name=f"pysnptools:read_plink:{path}",
    )

    # If contigs are already integers, use them as-is
    if bim_int_contig:
        variant_contig = arr_bim["contig"].astype("int16")
        variant_contig_names = da.unique(variant_contig).astype(str)
        variant_contig_names = list(variant_contig_names.compute())
    # Otherwise index by unique name where index will correspond
    # to lexsort on names, i.e. if contigs are 'chr1', 'chr2',
    # ..., 'chr10' then 'chr10' comes before 'chr2'
    else:
        variant_contig_names, variant_contig = da.unique(
            arr_bim["contig"], return_inverse=True
        )
        variant_contig_names = list(variant_contig_names.compute())
        variant_contig = variant_contig.astype("int16")

    variant_position = arr_bim["pos"]
    a1 = arr_bim["a1"].astype("str")
    a2 = arr_bim["a2"].astype("str")

    # Note: column_stack not implemented in Dask, must use [v|h]stack
    variant_alleles = da.hstack((a1[:, np.newaxis], a2[:, np.newaxis]))
    # TODO: Why use bytes for this instead of string?
    variant_alleles = variant_alleles.astype("S")
    variant_id = arr_bim["variant_id"]

    sample_id = arr_fam["member_id"]

    ds = create_genotype_call_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_alleles=variant_alleles,
        sample_id=sample_id,
        call_genotype=call_genotype,
        variant_id=variant_id,
    )

    # Assign PLINK-specific pedigree fields
    ds = ds.assign(**{f"sample/{f}": (DIM_SAMPLE, arr_fam[f]) for f in arr_fam})
    return ds