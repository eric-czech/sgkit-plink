"""PLINK 1.9 reader implementation"""
from pathlib import Path
from typing import Tuple, Union

import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.dataframe import DataFrame
from pysnptools.snpreader import Bed
from xarray import Dataset

from sgkit import create_genotype_call_dataset
from sgkit.api import DIM_SAMPLE
from sgkit.utils import encode_array

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
        # Slice using reversal of first two slices since
        # pysnptools uses sample x variant orientation.
        # Missing values are represented as -127 with int8 dtype,
        # see: https://fastlmm.github.io/PySnpTools/#snpreader-bed
        arr = (
            self.bed[idx[1::-1]]
            .read(dtype=np.int8, view_ok=True, _require_float32_64=False)
            .val.T
        )
        arr = arr.astype(self.dtype)
        # Add a ploidy dimension, so allele counts of 0, 1, 2 correspond to 00, 10, 11
        call0 = np.where(arr < 0, -1, np.where(arr == 0, 0, 1))
        call1 = np.where(arr < 0, -1, np.where(arr == 2, 1, 0))
        arr = np.stack([call0, call1], axis=-1)
        # Apply final slice to 3D result
        return arr[:, :, idx[-1]]

    def close(self):
        # This is not actually crucial since a Bed instance with no
        # in-memory bim/map/fam data is essentially just a file pointer
        # but this will still be problematic if the an array is created
        # from the same PLINK dataset many times
        self.bed._close_bed()  # pragma: no cover


def _to_dict(df, dtype=None):
    arrs = {}
    for c in df:
        a = df[c].to_dask_array(lengths=True)
        dt = df[c].dtype
        if dtype:
            dt = dtype[c]
        if dt in ["U", "S"]:
            # Compute fixed-length string dtype for array
            dt = a.astype(dt).compute().dtype
        arrs[c] = a.astype(dt)
    return arrs


def read_fam(path: PathType, sep: str = " ") -> DataFrame:
    # See: https://www.cog-genomics.org/plink/1.9/formats#fam
    names = [f[0] for f in FAM_FIELDS]
    df = dd.read_csv(str(path), sep=sep, names=names, dtype=FAM_DF_DTYPE)

    def coerce_code(v, codes):
        # Set non-ints and unexpected codes to missing (-1)
        v = dd.to_numeric(v, errors="coerce")
        v = v.where(v.isin(codes), np.nan)
        return v.fillna(-1).astype("int8")

    df["paternal_id"] = df["paternal_id"].where(df["paternal_id"] != "0", "")
    df["maternal_id"] = df["maternal_id"].where(df["maternal_id"] != "0", "")
    df["sex"] = coerce_code(df["sex"], [1, 2])
    df["phenotype"] = coerce_code(df["phenotype"], [1, 2])

    return df


def read_bim(path: PathType, sep: str = "\t") -> DataFrame:
    # See: https://www.cog-genomics.org/plink/1.9/formats#bim
    names = [f[0] for f in BIM_FIELDS]
    df = dd.read_csv(str(path), sep=sep, names=names, dtype=BIM_DF_DTYPE)
    df["contig"] = df["contig"].where(df["contig"] != "0", "")
    return df


def read_plink(
    path: Union[PathType, Tuple[PathType, PathType, PathType]],
    chunks: Union[str, int, tuple] = "auto",
    fam_sep: str = " ",
    bim_sep: str = "\t",
    bim_int_contig: bool = False,
    count_a1: bool = True,
    lock: bool = False,
    persist: bool = True,
) -> Dataset:
    """Read PLINK dataset.

    Loads a single PLINK dataset as dask arrays within a Dataset
    from bed, bim, and fam files.

    Parameters
    ----------
    path : Union[PathType, Tuple[PathType, PathType, PathType]]
        Path to PLINK file set.  This can either be one path or 3.
        If one path, then that path should not include a suffix such
        that if the files are at `data.{bed,bim,fam}` then only 'data'
        should be provided (suffixes are added internally). If 3 paths,
        then the paths must correspond to the `bed` file, the `bim`
        file and the `fam` file in that order (with extensions).
    chunks : Union[str, int, tuple], optional
        Chunk size for genotype (i.e. `.bed`) data, by default "auto"
    fam_sep : str, optional
        Delimiter for `.fam` file, by default " "
    bim_sep : str, optional
        Delimiter for `.bim` file, by default "\t"
    bim_int_contig : bool, optional
        Whether or not the contig/chromosome name in the `.bim`
        file should be interpreted as an integer, by default False.
        If False, then the `variant/contig` field in the resulting
        dataset will contain the indexes of corresponding strings
        encountered in the first `.bim` field.
    count_a1 : bool, optional
        Whether or not allele counts should be for A1 or A2,
        by default True. Typically A1 is the minor allele
        and should be counted instead of A2. This is not enforced
        by PLINK though and it is up to the data generating process
        to ensure that A1 is in fact an alternate/minor/effect
        allele. See https://www.cog-genomics.org/plink/1.9/formats
        for more details.
    lock : bool, optional
        Whether or not to synchronize concurrent reads of `.bed`
        file blocks, by default False. This is passed through to
        [dask.array.from_array](https://docs.dask.org/en/latest/array-api.html#dask.array.from_array).
    persist : bool, optional
        Whether or not to persist `.fam` and `.bim` information in
        memory, by default True. This is an important performance
        consideration as the plain text files for this data will
        be read multiple times when False. This can lead to load
        times that are upwards of 10x slower.

    Returns
    -------
    Dataset
        A dataset containing genotypes as 3 dimensional calls along with
        all accompanying pedigree and variant information. The content
        of this dataset matches that of `sgkit.create_genotype_call_dataset`
        with all pedigree-specific fields defined as:
            - sample/family_id: Family identifier commonly referred to as FID
            - sample/id: Within-family identifier for sample
            - sample/paternal_id: Within-family identifier for father of sample
            - sample/maternal_id: Within-family identifier for mother of sample
            - sample/sex: Sex code equal to 1 for male, 2 for female, and -1
                for missing
            - sample/phenotype: Phenotype code equal to 1 for control, 2 for case,
                and -1 for missing

        See https://www.cog-genomics.org/plink/1.9/formats#fam for more details.
    """
    if not isinstance(path, tuple):
        path = tuple(Path(path).with_suffix(ext) for ext in [".bed", ".bim", ".fam"])
    bed_path, bim_path, fam_path = path

    # Load axis data first to determine dimension sizes
    df_fam = read_fam(fam_path, sep=fam_sep)
    df_bim = read_bim(bim_path, sep=bim_sep)

    if persist:
        df_fam = df_fam.persist()
        df_bim = df_bim.persist()

    arr_fam = _to_dict(df_fam, dtype=FAM_ARRAY_DTYPE)
    arr_bim = _to_dict(df_bim, dtype=BIM_ARRAY_DTYPE)

    # Load genotyping data
    call_genotype = da.from_array(
        # Make sure to use asarray=False in order for masked arrays to propagate
        BedReader(bed_path, (len(df_bim), len(df_fam)), count_A1=count_a1),
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
    # Otherwise create index for contig names based
    # on order of appearance in underlying .bim file
    else:
        variant_contig, variant_contig_names = encode_array(arr_bim["contig"].compute())
        variant_contig = variant_contig.astype("int16")
        variant_contig_names = list(variant_contig_names)

    variant_position = arr_bim["pos"]
    a1 = arr_bim["a1"].astype("str")
    a2 = arr_bim["a2"].astype("str")

    # Note: column_stack not implemented in Dask, must use [v|h]stack
    variant_alleles = da.hstack((a1[:, np.newaxis], a2[:, np.newaxis]))
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
    ds = ds.assign(
        **{f"sample/{f}": (DIM_SAMPLE, arr_fam[f]) for f in arr_fam if f != "member_id"}
    )
    return ds
