"""
BMRB REST API v2 Client
========================

Fetches chemical shift assignments, relaxation data, J-coupling constants,
and associated PDB structures from the Biological Magnetic Resonance Bank.

API docs: https://github.com/bmrb-io/BMRB-API
Base URL: https://api.bmrb.io/v2/
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BMRB_API_BASE = "https://api.bmrb.io/v2"
BMRB_ENTRY_URL = "https://bmrb.io/ftp/pub/bmrb/entry_directories"

# Standard amino acid chemical shift statistics (mean, std) for each nucleus
# Used as priors in the assignment model. Source: BMRB statistics.
AA_SHIFT_STATS: dict[str, dict[str, tuple[float, float]]] = {
    "ALA": {"H": (8.19, 0.57), "N": (123.0, 3.7), "CA": (53.2, 1.9), "CB": (18.9, 1.8)},
    "ARG": {"H": (8.22, 0.59), "N": (120.6, 3.5), "CA": (56.8, 2.2), "CB": (30.6, 1.8)},
    "ASN": {"H": (8.33, 0.62), "N": (118.7, 3.9), "CA": (53.5, 1.9), "CB": (38.6, 1.7)},
    "ASP": {"H": (8.31, 0.57), "N": (120.6, 3.4), "CA": (54.7, 2.0), "CB": (40.8, 1.3)},
    "CYS": {"H": (8.32, 0.65), "N": (118.8, 4.2), "CA": (58.1, 3.1), "CB": (33.1, 5.0)},
    "GLN": {"H": (8.20, 0.58), "N": (119.8, 3.6), "CA": (56.6, 2.1), "CB": (29.1, 1.6)},
    "GLU": {"H": (8.33, 0.56), "N": (121.3, 3.3), "CA": (57.4, 2.0), "CB": (30.0, 1.5)},
    "GLY": {"H": (8.33, 0.64), "N": (109.5, 3.6), "CA": (45.3, 1.4), "CB": (0.0, 0.0)},
    "HIS": {"H": (8.26, 0.63), "N": (118.2, 4.0), "CA": (56.5, 2.4), "CB": (29.9, 2.0)},
    "ILE": {"H": (8.13, 0.62), "N": (121.4, 4.1), "CA": (61.6, 2.6), "CB": (38.5, 2.1)},
    "LEU": {"H": (8.17, 0.57), "N": (122.5, 3.5), "CA": (55.7, 2.0), "CB": (42.2, 1.8)},
    "LYS": {"H": (8.19, 0.56), "N": (121.2, 3.3), "CA": (57.0, 2.1), "CB": (32.7, 1.6)},
    "MET": {"H": (8.24, 0.57), "N": (120.0, 3.5), "CA": (56.2, 2.1), "CB": (32.9, 2.3)},
    "PHE": {"H": (8.24, 0.62), "N": (120.4, 3.8), "CA": (58.1, 2.4), "CB": (39.8, 1.9)},
    "PRO": {"H": (0.0, 0.0), "N": (136.7, 4.3), "CA": (63.3, 1.6), "CB": (31.8, 1.1)},
    "SER": {"H": (8.27, 0.57), "N": (116.0, 3.4), "CA": (58.7, 2.1), "CB": (63.6, 1.2)},
    "THR": {"H": (8.13, 0.60), "N": (114.2, 4.0), "CA": (62.2, 2.7), "CB": (69.6, 1.5)},
    "TRP": {"H": (8.15, 0.67), "N": (121.4, 3.8), "CA": (57.7, 2.5), "CB": (29.6, 1.9)},
    "TYR": {"H": (8.14, 0.62), "N": (120.2, 3.7), "CA": (58.1, 2.5), "CB": (38.9, 1.9)},
    "VAL": {"H": (8.10, 0.62), "N": (121.0, 4.2), "CA": (62.5, 2.9), "CB": (32.5, 1.7)},
}

# Three-letter to one-letter amino acid mapping
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


class BMRBClient:
    """Client for the BMRB REST API v2.

    Fetches chemical shift assignments, relaxation data, J-couplings,
    and associated structural metadata from the BMRB database.

    Implements local caching and rate limiting to be a good API citizen.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for caching downloaded entries.
    rate_limit : float
        Minimum seconds between API requests.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/bmrb_cache",
        rate_limit: float = 0.5,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self._last_request_time: float = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "Application": "NMR-Physics-Pipeline/0.1",
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _get(self, url: str, params: dict | None = None) -> dict[str, Any]:
        """Make a rate-limited GET request with error handling."""
        self._rate_limit_wait()
        try:
            resp = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error("BMRB API request failed: %s — %s", url, e)
            raise

    def _cache_path(self, entry_id: str, suffix: str = "json") -> Path:
        """Return cache file path for an entry."""
        return self.cache_dir / f"{entry_id}_{suffix}.json"

    def _load_cached(self, entry_id: str, suffix: str = "json") -> dict | None:
        """Load cached data if available."""
        path = self._cache_path(entry_id, suffix)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, entry_id: str, data: dict, suffix: str = "json") -> None:
        """Save data to local cache."""
        path = self._cache_path(entry_id, suffix)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Public API — Entry-level access
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> dict[str, Any]:
        """Fetch a complete BMRB entry as JSON.

        Parameters
        ----------
        entry_id : str
            BMRB entry ID (e.g., '4493' for ubiquitin).

        Returns
        -------
        dict
            Full entry data in JSON format.
        """
        cached = self._load_cached(entry_id, "full")
        if cached is not None:
            logger.info("Loaded entry %s from cache", entry_id)
            return cached

        url = f"{BMRB_API_BASE}/entry/{entry_id}"
        data = self._get(url)
        self._save_cache(entry_id, data, "full")
        return data

    def get_entry_metadata(self, entry_id: str) -> dict[str, Any]:
        """Fetch metadata for an entry (title, authors, related PDB IDs).

        Returns
        -------
        dict with keys: title, authors, pdb_ids, citation, molecular_system
        """
        url = f"{BMRB_API_BASE}/entry/{entry_id}"
        params = {
            "saveframe_category": "entry_information",
            "format": "json",
        }
        return self._get(url, params)

    # ------------------------------------------------------------------
    # Chemical Shift Data
    # ------------------------------------------------------------------

    def get_chemical_shifts(self, entry_id: str) -> pd.DataFrame:
        """Fetch assigned chemical shifts for a BMRB entry.

        Returns a DataFrame with columns:
            seq_id, comp_id (residue type), atom_id, atom_type, shift_value, shift_error

        Parameters
        ----------
        entry_id : str
            BMRB entry ID.

        Returns
        -------
        pd.DataFrame
            Chemical shift assignments table.
        """
        cached = self._load_cached(entry_id, "shifts")
        if cached is not None:
            return pd.DataFrame(cached)

        url = f"{BMRB_API_BASE}/entry/{entry_id}"
        params = {"saveframe_category": "assigned_chemical_shifts"}
        data = self._get(url, params)

        # Parse the NMR-STAR response into a flat table
        shifts = self._parse_chemical_shifts(data, entry_id)
        if shifts is not None and len(shifts) > 0:
            self._save_cache(entry_id, shifts.to_dict(orient="records"), "shifts")
        return shifts

    def _parse_chemical_shifts(self, data: dict, entry_id: str) -> pd.DataFrame:
        """Parse chemical shift data from BMRB JSON response.

        Handles the nested NMR-STAR structure to extract a flat table.
        """
        records = []

        try:
            # Navigate the NMR-STAR JSON structure
            entry_data = data.get(entry_id, data)

            # Try to find the assigned_chem_shift_list saveframe
            saveframes = []
            if isinstance(entry_data, dict):
                for key, val in entry_data.items():
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict):
                                sf_cat = item.get("sf_category", "")
                                if "assigned_chem" in sf_cat.lower():
                                    saveframes.append(item)

            if not saveframes:
                # Try alternative structure: direct loop access
                loops = entry_data if isinstance(entry_data, list) else [entry_data]
                for loop in loops:
                    if isinstance(loop, dict) and "Atom_chem_shift" in str(loop):
                        saveframes.append(loop)

            for sf in saveframes:
                # Look for the Atom_chem_shift loop
                loops = sf.get("loops", [])
                if not loops and isinstance(sf, dict):
                    # Try flat structure
                    loops = [sf]

                for loop in loops:
                    tags = loop.get("tags", loop.get("columns", []))
                    data_rows = loop.get("data", loop.get("rows", []))

                    if not tags or not data_rows:
                        continue

                    # Map tag names to indices
                    tag_map = {}
                    for i, tag in enumerate(tags):
                        tag_lower = tag.lower().replace(".", "_")
                        if "seq_id" in tag_lower or "comp_index_id" in tag_lower:
                            tag_map["seq_id"] = i
                        elif "comp_id" in tag_lower:
                            tag_map["comp_id"] = i
                        elif "atom_id" in tag_lower and "atom_type" not in tag_lower:
                            tag_map["atom_id"] = i
                        elif "atom_type" in tag_lower:
                            tag_map["atom_type"] = i
                        elif "val" in tag_lower and "err" not in tag_lower:
                            tag_map["shift_value"] = i
                        elif "val_err" in tag_lower or "error" in tag_lower:
                            tag_map["shift_error"] = i

                    if "shift_value" not in tag_map:
                        continue

                    for row in data_rows:
                        if not isinstance(row, (list, tuple)):
                            continue
                        record = {}
                        for field, idx in tag_map.items():
                            if idx < len(row):
                                val = row[idx]
                                if val in (".", "?", None):
                                    val = None
                                record[field] = val
                        if record.get("shift_value") is not None:
                            try:
                                record["shift_value"] = float(record["shift_value"])
                            except (ValueError, TypeError):
                                continue
                            if record.get("shift_error") is not None:
                                try:
                                    record["shift_error"] = float(record["shift_error"])
                                except (ValueError, TypeError):
                                    record["shift_error"] = None
                            if record.get("seq_id") is not None:
                                try:
                                    record["seq_id"] = int(record["seq_id"])
                                except (ValueError, TypeError):
                                    pass
                            records.append(record)

        except Exception as e:
            logger.warning("Failed to parse shifts for entry %s: %s", entry_id, e)

        if not records:
            logger.warning("No chemical shift data found for entry %s", entry_id)
            return pd.DataFrame(
                columns=["seq_id", "comp_id", "atom_id", "atom_type", "shift_value", "shift_error"]
            )

        df = pd.DataFrame(records)
        # Ensure expected columns exist
        for col in ["seq_id", "comp_id", "atom_id", "atom_type", "shift_value", "shift_error"]:
            if col not in df.columns:
                df[col] = None

        return df[["seq_id", "comp_id", "atom_id", "atom_type", "shift_value", "shift_error"]]

    # ------------------------------------------------------------------
    # Relaxation Data
    # ------------------------------------------------------------------

    def get_relaxation_data(self, entry_id: str) -> pd.DataFrame:
        """Fetch relaxation data (T1, T2, heteronuclear NOE) for an entry.

        Returns
        -------
        pd.DataFrame
            Relaxation measurements with columns:
            seq_id, comp_id, atom_id, relaxation_type, value, error, field_strength
        """
        cached = self._load_cached(entry_id, "relaxation")
        if cached is not None:
            return pd.DataFrame(cached)

        url = f"{BMRB_API_BASE}/entry/{entry_id}"
        params = {"saveframe_category": "heteronucl_T1_relaxation,heteronucl_T2_relaxation,heteronucl_NOEs"}
        try:
            data = self._get(url, params)
        except requests.exceptions.HTTPError:
            logger.info("No relaxation data for entry %s", entry_id)
            return pd.DataFrame(
                columns=["seq_id", "comp_id", "atom_id", "relaxation_type",
                         "value", "error", "field_strength"]
            )

        records = self._parse_relaxation(data, entry_id)
        df = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["seq_id", "comp_id", "atom_id", "relaxation_type",
                     "value", "error", "field_strength"]
        )
        if len(df) > 0:
            self._save_cache(entry_id, df.to_dict(orient="records"), "relaxation")
        return df

    def _parse_relaxation(self, data: dict, entry_id: str) -> list[dict]:
        """Parse relaxation data from BMRB JSON response."""
        records = []
        # Relaxation data follows similar NMR-STAR structure
        # Each saveframe has a type (T1, T2, NOE) and a data loop
        try:
            entry_data = data.get(entry_id, data)
            if isinstance(entry_data, dict):
                for _key, val in entry_data.items():
                    if isinstance(val, list):
                        for item in val:
                            if not isinstance(item, dict):
                                continue
                            sf_cat = item.get("sf_category", "").lower()
                            if "t1" in sf_cat:
                                relax_type = "T1"
                            elif "t2" in sf_cat:
                                relax_type = "T2"
                            elif "noe" in sf_cat:
                                relax_type = "hetNOE"
                            else:
                                continue

                            for loop in item.get("loops", []):
                                tags = loop.get("tags", [])
                                rows = loop.get("data", [])
                                tag_map = self._build_relaxation_tag_map(tags)
                                for row in rows:
                                    if not isinstance(row, (list, tuple)):
                                        continue
                                    rec = self._extract_relaxation_record(
                                        row, tag_map, relax_type
                                    )
                                    if rec is not None:
                                        records.append(rec)
        except Exception as e:
            logger.warning("Failed to parse relaxation for %s: %s", entry_id, e)

        return records

    @staticmethod
    def _build_relaxation_tag_map(tags: list[str]) -> dict[str, int]:
        """Build column index map for relaxation data loop."""
        tag_map = {}
        for i, tag in enumerate(tags):
            tl = tag.lower()
            if "seq_id" in tl or "comp_index" in tl:
                tag_map["seq_id"] = i
            elif "comp_id" in tl:
                tag_map["comp_id"] = i
            elif "atom_id" in tl:
                tag_map["atom_id"] = i
            elif "val" in tl and "err" not in tl:
                tag_map["value"] = i
            elif "err" in tl:
                tag_map["error"] = i
            elif "field_strength" in tl or "spectrometer_freq" in tl:
                tag_map["field_strength"] = i
        return tag_map

    @staticmethod
    def _extract_relaxation_record(
        row: list, tag_map: dict[str, int], relax_type: str
    ) -> dict | None:
        """Extract a single relaxation record from a data row."""
        if "value" not in tag_map:
            return None
        try:
            val = float(row[tag_map["value"]])
        except (ValueError, TypeError, IndexError):
            return None

        rec = {"relaxation_type": relax_type, "value": val}
        for field in ["seq_id", "comp_id", "atom_id", "error", "field_strength"]:
            idx = tag_map.get(field)
            if idx is not None and idx < len(row):
                v = row[idx]
                if v in (".", "?", None):
                    rec[field] = None
                else:
                    try:
                        rec[field] = float(v) if field in ("error", "field_strength") else v
                    except (ValueError, TypeError):
                        rec[field] = v
            else:
                rec[field] = None
        if rec.get("seq_id") is not None:
            try:
                rec["seq_id"] = int(rec["seq_id"])
            except (ValueError, TypeError):
                pass
        return rec

    # ------------------------------------------------------------------
    # J-Coupling Data
    # ------------------------------------------------------------------

    def get_j_couplings(self, entry_id: str) -> pd.DataFrame:
        """Fetch J-coupling constants for an entry.

        Returns
        -------
        pd.DataFrame
            Columns: seq_id_1, atom_id_1, seq_id_2, atom_id_2, coupling_value, error
        """
        cached = self._load_cached(entry_id, "jcouplings")
        if cached is not None:
            return pd.DataFrame(cached)

        url = f"{BMRB_API_BASE}/entry/{entry_id}"
        params = {"saveframe_category": "coupling_constants"}
        try:
            data = self._get(url, params)
        except requests.exceptions.HTTPError:
            return pd.DataFrame(
                columns=["seq_id_1", "atom_id_1", "seq_id_2", "atom_id_2",
                         "coupling_value", "error"]
            )

        # J-coupling parsing follows similar patterns
        records = []
        try:
            entry_data = data.get(entry_id, data)
            if isinstance(entry_data, dict):
                for _key, val in entry_data.items():
                    if isinstance(val, list):
                        for item in val:
                            if not isinstance(item, dict):
                                continue
                            for loop in item.get("loops", []):
                                tags = loop.get("tags", [])
                                rows = loop.get("data", [])
                                for row in rows:
                                    rec = self._parse_jcoupling_row(tags, row)
                                    if rec is not None:
                                        records.append(rec)
        except Exception as e:
            logger.warning("Failed to parse J-couplings for %s: %s", entry_id, e)

        df = pd.DataFrame(records) if records else pd.DataFrame(
            columns=["seq_id_1", "atom_id_1", "seq_id_2", "atom_id_2",
                     "coupling_value", "error"]
        )
        if len(df) > 0:
            self._save_cache(entry_id, df.to_dict(orient="records"), "jcouplings")
        return df

    @staticmethod
    def _parse_jcoupling_row(tags: list[str], row: list) -> dict | None:
        """Parse a single J-coupling row."""
        tag_map = {}
        for i, tag in enumerate(tags):
            tl = tag.lower()
            if "seq_id_1" in tl:
                tag_map["seq_id_1"] = i
            elif "seq_id_2" in tl:
                tag_map["seq_id_2"] = i
            elif "atom_id_1" in tl:
                tag_map["atom_id_1"] = i
            elif "atom_id_2" in tl:
                tag_map["atom_id_2"] = i
            elif "val" in tl and "err" not in tl:
                tag_map["coupling_value"] = i
            elif "err" in tl:
                tag_map["error"] = i

        if "coupling_value" not in tag_map:
            return None
        try:
            rec = {"coupling_value": float(row[tag_map["coupling_value"]])}
        except (ValueError, TypeError, IndexError):
            return None

        for field in ["seq_id_1", "atom_id_1", "seq_id_2", "atom_id_2", "error"]:
            idx = tag_map.get(field)
            if idx is not None and idx < len(row):
                v = row[idx]
                rec[field] = v if v not in (".", "?", None) else None
            else:
                rec[field] = None
        return rec

    # ------------------------------------------------------------------
    # Search and Discovery
    # ------------------------------------------------------------------

    def search_by_sequence(
        self, sequence: str, database: str = "macromolecules"
    ) -> list[dict]:
        """Search BMRB entries by protein sequence.

        Parameters
        ----------
        sequence : str
            One-letter amino acid sequence.
        database : str
            Database to search ('macromolecules' or 'metabolomics').

        Returns
        -------
        list[dict]
            List of matching entries with entry_id, title, organism.
        """
        url = f"{BMRB_API_BASE}/search/chemical_shifts"
        params = {"comp_id": sequence[:3]}  # Simplified — full sequence search not in v2
        # Use the instant search for broader matching
        url = f"{BMRB_API_BASE}/instant"
        params = {"term": sequence[:20], "database": database}
        try:
            data = self._get(url, params)
            return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.warning("Sequence search failed: %s", e)
            return []

    def search_small_molecules(self, name: str) -> list[dict]:
        """Search BMRB metabolomics database for small molecules.

        Parameters
        ----------
        name : str
            Molecule name or partial name.

        Returns
        -------
        list[dict]
            Matching metabolomics entries.
        """
        url = f"{BMRB_API_BASE}/instant"
        params = {"term": name, "database": "metabolomics"}
        try:
            data = self._get(url, params)
            return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.warning("Small molecule search failed: %s", e)
            return []

    def get_associated_pdb_ids(self, entry_id: str) -> list[str]:
        """Get PDB IDs associated with a BMRB entry.

        Returns
        -------
        list[str]
            Associated PDB codes (e.g., ['1UBQ']).
        """
        try:
            meta = self.get_entry_metadata(entry_id)
            # Extract PDB IDs from the metadata
            pdb_ids = []
            if isinstance(meta, dict):
                entry_data = meta.get(entry_id, meta)
                # Search for PDB cross-references
                self._extract_pdb_ids_recursive(entry_data, pdb_ids)
            return list(set(pdb_ids))
        except Exception as e:
            logger.warning("Failed to get PDB IDs for %s: %s", entry_id, e)
            return []

    @staticmethod
    def _extract_pdb_ids_recursive(obj: Any, pdb_ids: list[str]) -> None:
        """Recursively search JSON for PDB ID references."""
        if isinstance(obj, dict):
            for key, val in obj.items():
                if "pdb" in key.lower() and isinstance(val, str) and len(val) == 4:
                    pdb_ids.append(val.upper())
                else:
                    BMRBClient._extract_pdb_ids_recursive(val, pdb_ids)
        elif isinstance(obj, list):
            for item in obj:
                BMRBClient._extract_pdb_ids_recursive(item, pdb_ids)
        elif isinstance(obj, str) and len(obj) == 4 and obj[0].isdigit():
            # Heuristic: 4-character strings starting with digit might be PDB IDs
            pass  # Too aggressive — skip to avoid false positives

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def get_entry_list(self, database: str = "macromolecules") -> list[str]:
        """Get list of all available BMRB entry IDs.

        Parameters
        ----------
        database : str
            'macromolecules' or 'metabolomics'.

        Returns
        -------
        list[str]
            All available entry IDs.
        """
        url = f"{BMRB_API_BASE}/list_entries"
        params = {"database": database}
        try:
            data = self._get(url, params)
            if isinstance(data, list):
                return [str(e) for e in data]
            return data.get("data", [])
        except Exception as e:
            logger.warning("Failed to get entry list: %s", e)
            return []

    def build_training_dataset(
        self,
        entry_ids: list[str] | None = None,
        max_entries: int = 100,
        min_shifts: int = 10,
    ) -> list[dict]:
        """Build a training dataset from BMRB entries.

        Fetches chemical shifts for each entry and returns a list of
        training examples with sequence, shifts, and metadata.

        Parameters
        ----------
        entry_ids : list[str] or None
            Specific entries to fetch. If None, auto-selects entries.
        max_entries : int
            Maximum number of entries to process.
        min_shifts : int
            Minimum number of shift assignments to include an entry.

        Returns
        -------
        list[dict]
            Training examples with keys:
            entry_id, sequence, shifts (DataFrame), relaxation (DataFrame)
        """
        if entry_ids is None:
            all_ids = self.get_entry_list()
            # Sample entries — prefer smaller entry IDs (earlier, better curated)
            entry_ids = sorted(all_ids)[:max_entries]

        dataset = []
        for eid in entry_ids[:max_entries]:
            logger.info("Processing BMRB entry %s (%d/%d)", eid, len(dataset) + 1, max_entries)
            try:
                shifts = self.get_chemical_shifts(eid)
                if shifts is None or len(shifts) < min_shifts:
                    continue

                # Extract sequence from shift assignments
                sequence = self._extract_sequence(shifts)
                if not sequence:
                    continue

                # Also try to get relaxation data
                relaxation = self.get_relaxation_data(eid)

                dataset.append({
                    "entry_id": eid,
                    "sequence": sequence,
                    "shifts": shifts,
                    "relaxation": relaxation,
                })

            except Exception as e:
                logger.warning("Failed to process entry %s: %s", eid, e)
                continue

        logger.info("Built dataset with %d entries from BMRB", len(dataset))
        return dataset

    @staticmethod
    def _extract_sequence(shifts_df: pd.DataFrame) -> str:
        """Extract protein sequence from chemical shift assignments."""
        if "seq_id" not in shifts_df.columns or "comp_id" not in shifts_df.columns:
            return ""

        # Get unique residues ordered by sequence ID
        residues = (
            shifts_df[["seq_id", "comp_id"]]
            .drop_duplicates()
            .sort_values("seq_id")
        )

        sequence = []
        for _, row in residues.iterrows():
            aa3 = str(row["comp_id"]).upper().strip()
            aa1 = AA_3TO1.get(aa3, "X")
            sequence.append(aa1)

        return "".join(sequence)

    # ------------------------------------------------------------------
    # Chemical shift statistics
    # ------------------------------------------------------------------

    @staticmethod
    def get_shift_statistics() -> dict[str, dict[str, tuple[float, float]]]:
        """Return amino acid chemical shift statistics (mean, std).

        Returns the BMRB-derived statistical distributions for backbone
        chemical shifts, useful as priors in the assignment model.

        Returns
        -------
        dict
            Nested dict: AA_3letter -> nucleus -> (mean, std)
        """
        return AA_SHIFT_STATS.copy()

    @staticmethod
    def shift_probability(
        shift_value: float,
        residue_type: str,
        nucleus: str,
    ) -> float:
        """Calculate probability of a chemical shift given residue type.

        Uses Gaussian distribution based on BMRB statistics.

        Parameters
        ----------
        shift_value : float
            Observed chemical shift in ppm.
        residue_type : str
            Three-letter amino acid code.
        nucleus : str
            Nucleus type ('H', 'N', 'CA', 'CB').

        Returns
        -------
        float
            Log-probability of the shift given the residue type.
        """
        stats = AA_SHIFT_STATS.get(residue_type.upper())
        if stats is None:
            return -10.0  # Unknown residue type

        nuc_stats = stats.get(nucleus.upper())
        if nuc_stats is None or nuc_stats[1] == 0.0:
            return -10.0  # Unknown nucleus or zero std

        mean, std = nuc_stats
        z = (shift_value - mean) / std
        # Log of Gaussian PDF (up to constant)
        return -0.5 * z * z - np.log(std)
