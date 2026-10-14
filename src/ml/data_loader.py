import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
import pandas as pd


def _col_to_num(col_str: str) -> int:
    num = 0
    for char in col_str:
        num = num * 26 + (ord(char.upper()) - ord("A")) + 1
    return num - 1


def _parse_xlsx_pure(file_path: Path) -> pd.DataFrame:
    """Parse .xlsx without external dependencies using zipfile and ElementTree."""
    with zipfile.ZipFile(file_path, "r") as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in tree.findall("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                texts = [
                    t.text for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                    if t.text is not None
                ]
                shared_strings.append("".join(texts))

        sheet_tree = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        sheet_data = sheet_tree.find(f"{ns}sheetData")

        matrix = []
        for row_el in sheet_data.findall(f"{ns}row"):
            row_idx = int(row_el.get("r")) - 1
            row_vals = {}
            max_col = 0
            for c in row_el.findall(f"{ns}c"):
                r = c.get("r")
                match = re.match(r"([A-Z]+)", r)
                col_letters = match.group(1) if match else "A"
                c_idx = _col_to_num(col_letters)
                max_col = max(max_col, c_idx)

                t = c.get("t")
                v = c.find(f"{ns}v")
                val = v.text if v is not None else None
                if val is not None:
                    if t == "s":
                        val = shared_strings[int(val)]
                    elif t == "b":
                        val = bool(int(val))
                row_vals[c_idx] = val
            matrix.append((row_idx, row_vals, max_col))

    num_cols = max(m[2] for m in matrix) + 1
    data = []
    for _, r_dict, _ in matrix:
        row = [r_dict.get(c, None) for c in range(num_cols)]
        data.append(row)

    header = data[0]
    rows = data[1:]
    return pd.DataFrame(rows, columns=header)


def load_raw_dataset(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Z-Alizadeh Sani extension dataset.

    Attempts openpyxl engine first, falling back to pure XML parsing.
    """
    if file_path is None:
        candidates = [
            Path("data/raw/extention of Z-Alizadeh sani dataset.xlsx"),
            Path("data/raw/extention_of_z_alizadeh_sani_dataset.xlsx"),
            Path("extention+of+z+alizadeh+sani+dataset/extention of Z-Alizadeh sani dataset.xlsx"),
        ]
        target_path = next((p for p in candidates if p.exists()), None)
        if target_path is None:
            raise FileNotFoundError("Raw dataset not found in data/raw/ or project directory.")
    else:
        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Specified dataset file not found: {target_path}")

    try:
        import openpyxl  # noqa: F401
        return pd.read_excel(target_path, engine="openpyxl")
    except (ImportError, Exception):
        return _parse_xlsx_pure(target_path)
