"""Small, Spark-first helpers used by the ENEIC Personas laboratory.

The functions in this module deliberately read one Excel file at a time and
only read the selected columns.  After that boundary, calculations and data
quality checks stay in Spark.  In particular, no helper combines 2025 and
2026: the period is derived from each file name and attached as metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from numbers import Real
from pathlib import Path
import math
import re
import unicodedata
from typing import Any


SEED = 42

# Source fields retained from the otherwise wide ENEIC Personas workbooks.
REQUIRED_PERSON_COLUMNS: tuple[str, ...] = (
    "ANIO",
    "TRIMESTRE",
    "DOMINIO",
    "NUM_HOGAR",
    "FACTOR",
    "NUM_PERSONA",
    "P02A03",
    "P03A03A",
    "P05C07A",
    "P05C07B",
    "P05C16",
    "P05D01",
    "P05H01A",
    "OCUPADOS",
)

# ``dict`` order is intentional: it is also the standard selected schema.
STANDARD_COLUMN_MAP: dict[str, str] = {
    "P05D01": "salario_mensual",
    "P02A03": "edad",
    "P05C07A": "antiguedad_anios",
    "P05C07B": "antiguedad_meses",
    "P05H01A": "horas_semanales",
    "P03A03A": "nivel_educativo",
    "P05C16": "categoria_ocupacional",
    "DOMINIO": "dominio",
    "OCUPADOS": "ocupado",
    "NUM_HOGAR": "NUM_HOGAR",
    "NUM_PERSONA": "NUM_PERSONA",
    "FACTOR": "FACTOR",
    "ANIO": "ANIO",
    "TRIMESTRE": "TRIMESTRE",
}

SELECTED_SOURCE_COLUMNS: tuple[str, ...] = tuple(STANDARD_COLUMN_MAP)
# Short aliases make the constants convenient in notebook cells as well.
REQUIRED_COLUMNS = REQUIRED_PERSON_COLUMNS
SELECTED_COLUMNS = SELECTED_SOURCE_COLUMNS
STANDARD_ANALYTIC_COLUMNS: tuple[str, ...] = tuple(STANDARD_COLUMN_MAP.values()) + (
    "antiguedad",
    "archivo_origen",
    "periodo_archivo",
    "anio_archivo",
    "trimestre_calendario",
)
MISSINGNESS_COLUMNS: tuple[str, ...] = (
    "salario_mensual",
    "edad",
    "antiguedad_anios",
    "antiguedad_meses",
    "horas_semanales",
    "nivel_educativo",
    "categoria_ocupacional",
    "dominio",
    "ocupado",
    "NUM_HOGAR",
    "NUM_PERSONA",
    "FACTOR",
)
NUMERIC_ANALYSIS_COLUMNS: tuple[str, ...] = (
    "salario_mensual",
    "edad",
    "antiguedad",
    "horas_semanales",
)
KMEANS_FEATURE_COLUMNS: tuple[str, ...] = (
    "edad",
    "antiguedad",
    "horas_semanales",
)
SUPERVISED_NUMERIC_PREDICTORS: tuple[str, ...] = (
    "edad",
    "antiguedad",
    "horas_semanales",
)
SUPERVISED_CATEGORICAL_PREDICTORS: tuple[str, ...] = (
    "nivel_educativo",
    "categoria_ocupacional",
    "dominio",
)
SUPERVISED_PREDICTORS: tuple[str, ...] = (
    *SUPERVISED_NUMERIC_PREDICTORS,
    *SUPERVISED_CATEGORICAL_PREDICTORS,
)

EXCEL_SUFFIXES: tuple[str, ...] = (".xlsx", ".xlsm", ".xls")
VALID_OCCUPATIONAL_CATEGORIES: tuple[str, ...] = ("1", "2", "3", "4")
PERIOD_METADATA_COLUMNS: tuple[str, ...] = (
    "archivo_origen",
    "periodo_archivo",
    "anio_archivo",
    "trimestre_calendario",
)

_NUMERIC_SOURCE_COLUMNS = frozenset(
    {
        "P05D01",
        "P02A03",
        "P05C07A",
        "P05C07B",
        "P05H01A",
        "FACTOR",
    }
)
_CODE_SOURCE_COLUMNS = frozenset(
    {
        "P03A03A",
        "P05C16",
        "DOMINIO",
        "OCUPADOS",
        "NUM_HOGAR",
        "NUM_PERSONA",
        "ANIO",
        "TRIMESTRE",
    }
)
_ROMAN_QUARTERS = {"I": 1, "II": 2, "III": 3, "IV": 4}
_WORD_QUARTERS = {
    "PRIMER": 1,
    "PRIMERO": 1,
    "SEGUNDO": 2,
    "TERCER": 3,
    "TERCERO": 3,
    "CUARTO": 4,
}


class RequiredColumnsError(ValueError):
    """Raised when an Excel sheet does not contain the required ENEIC fields."""


def _normalise_text(value: object) -> str:
    """Return uppercase ASCII text suitable for tolerant file-name matching."""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.upper()


def _clean_header(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _as_existing_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"No se encontró el archivo Excel: {candidate}")
    return candidate


def _is_person_file(path: Path, suffixes: set[str]) -> bool:
    if path.suffix.lower() not in suffixes:
        return False
    name = _normalise_text(path.name)
    # Excluding the whole relative path also avoids a Personas-named file kept
    # in a Hogares/Vivienda folder by accident.
    full_name = _normalise_text(str(path))
    return (
        "PERSONA" in name
        and "HOGAR" not in full_name
        and "VIVIENDA" not in full_name
        and "DICCIONARIO" not in name
        and "DICTIONARY" not in name
        and "BOLETA" not in name
    )


def find_person_files(
    root: str | Path | Sequence[str | Path] = Path("data/raw"),
    *,
    recursive: bool = True,
    extensions: Sequence[str] = EXCEL_SUFFIXES,
) -> list[Path]:
    """Find ENEIC *Personas* Excel files while ignoring other modules.

    A missing directory simply returns an empty list, which lets a fresh clone
    of the repository run its setup cells without pretending that 2025 data is
    available.  Results are ordered by detected year/quarter when possible.
    """

    suffixes = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    candidates: list[Path] = []
    for root_item in roots:
        base = Path(root_item).expanduser()
        if base.is_file():
            candidates.append(base)
        elif base.exists():
            iterator = base.rglob("*") if recursive else base.glob("*")
            candidates.extend(item for item in iterator if item.is_file())

    matches = [item for item in candidates if _is_person_file(item, suffixes)]

    def sort_key(item: Path) -> tuple[int, int, str]:
        try:
            period = detect_period_from_filename(item)
            return (
                int(period["anio_archivo"]),
                int(period["trimestre_calendario"]),
                str(item).casefold(),
            )
        except ValueError:
            return (9999, 99, str(item).casefold())

    return sorted(matches, key=sort_key)


def detect_period_from_filename(path: str | Path) -> dict[str, int | str]:
    """Derive ENEIC period metadata from a file name, never from ``TRIMESTRE``.

    Roman (``I``--``IV``), explicit ``T1``/``Trimestre 1``, and common Spanish
    word variants are accepted.  The result has exactly the metadata used by
    loading code: ``periodo_archivo``, ``anio_archivo``, and
    ``trimestre_calendario``.  An ambiguous or unknown name raises ``ValueError``
    instead of guessing a period.
    """

    filename = Path(path).stem
    text = _normalise_text(filename)
    years = re.findall(r"(?<!\d)(20\d{2})(?!\d)", text)
    if not years:
        raise ValueError(
            f"No se pudo identificar el año en el nombre de archivo: {Path(path).name}"
        )
    if len(set(years)) != 1:
        raise ValueError(
            f"El nombre de archivo contiene años ambiguos ({', '.join(years)}): "
            f"{Path(path).name}"
        )
    year = int(years[0])

    quarters: list[int] = []
    # T1, trimestre 1, Q1, including compact forms such as 2025T1.
    for pattern in (
        r"(?:^|[^A-Z0-9])(?:T|TRIM|TRIMESTRE|Q)\s*[-_ ]*([1-4])(?:$|[^0-9])",
        r"20\d{2}\s*[-_ ]*(?:T|Q)\s*[-_ ]*([1-4])\b",
        r"\b([1-4])(?:ER|RO|DO|TO)?\s*[-_ ]*(?:TRIM|TRIMESTRE)\b",
    ):
        quarters.extend(int(match) for match in re.findall(pattern, text))

    tokens = re.findall(r"[A-Z0-9]+", text)
    for token in tokens:
        if token in {"T1", "T2", "T3", "T4", "Q1", "Q2", "Q3", "Q4"}:
            quarters.append(int(token[-1]))
        elif token in _WORD_QUARTERS:
            quarters.append(_WORD_QUARTERS[token])

    # In the usual ENEIC-I-2025 form, the Roman token immediately follows
    # ENEIC.  A nearby-year fallback covers Personas-I-2025 too.
    roman_candidates: list[int] = []
    for index, token in enumerate(tokens):
        if token == "ENEIC":
            for next_token in tokens[index + 1 : index + 5]:
                if next_token in _ROMAN_QUARTERS:
                    roman_candidates.append(_ROMAN_QUARTERS[next_token])
                    break
                if next_token.isdigit() and len(next_token) == 4:
                    break
    for index, token in enumerate(tokens):
        if token in _ROMAN_QUARTERS:
            nearby = tokens[max(0, index - 3) : index + 4]
            if str(year) in nearby:
                roman_candidates.append(_ROMAN_QUARTERS[token])

    all_candidates = set(quarters + roman_candidates)
    if not all_candidates:
        raise ValueError(
            "No se pudo identificar el trimestre (I-IV, T1-T4 o trimestre 1-4) "
            f"en: {Path(path).name}"
        )
    if len(all_candidates) != 1:
        values = ", ".join(str(value) for value in sorted(all_candidates))
        raise ValueError(
            f"El nombre de archivo contiene trimestres ambiguos ({values}): "
            f"{Path(path).name}"
        )

    quarter = all_candidates.pop()
    return {
        "periodo_archivo": f"{year}T{quarter}",
        "anio_archivo": year,
        "trimestre_calendario": quarter,
    }


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "La lectura inicial de Excel requiere pandas y openpyxl. "
            "Instálelos antes de ejecutar la carga."
        ) from exc
    return pd


def _read_excel_header_info(
    path: str | Path,
    sheet_name: str | int = 0,
    *,
    engine: str | None = "openpyxl",
) -> tuple[Any, list[object], list[str]]:
    excel_path = _as_existing_path(path)
    pd = _import_pandas()
    try:
        header = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            nrows=0,
            engine=engine,
        )
    except Exception as exc:
        raise ValueError(
            f"No se pudo leer el encabezado de '{excel_path.name}', hoja "
            f"{sheet_name!r}: {exc}"
        ) from exc

    raw_columns = list(header.columns)
    cleaned_columns = [_clean_header(column) for column in raw_columns]
    duplicates = sorted(
        {column for column in cleaned_columns if cleaned_columns.count(column) > 1}
    )
    if duplicates:
        raise ValueError(
            f"La hoja '{excel_path.name}' tiene encabezados duplicados después de "
            f"normalizar espacios: {duplicates}"
        )
    return pd, raw_columns, cleaned_columns


def read_excel_columns(
    path: str | Path,
    sheet_name: str | int = 0,
    *,
    engine: str | None = "openpyxl",
) -> list[str]:
    """Read only the Excel header and return clean column names.

    This is intentionally a cheap schema audit; it does not load the 270/302
    data columns into pandas.
    """

    _, _, columns = _read_excel_header_info(path, sheet_name, engine=engine)
    return columns


def _normalise_requested_columns(
    selected_columns: Sequence[str] | Mapping[str, str] | None,
) -> dict[str, str]:
    if selected_columns is None:
        return dict(STANDARD_COLUMN_MAP)
    if isinstance(selected_columns, Mapping):
        result = {
            _clean_header(source): _clean_header(target)
            for source, target in selected_columns.items()
        }
    else:
        result = {
            _clean_header(source): STANDARD_COLUMN_MAP.get(
                _clean_header(source), _clean_header(source)
            )
            for source in selected_columns
        }
    if not result:
        raise ValueError("selected_columns no puede estar vacío.")
    if len(set(result.values())) != len(result):
        raise ValueError("selected_columns asigna más de una columna al mismo nombre.")
    return result


def _schema_error(path: Path, missing: Sequence[str], available: Sequence[str]) -> RequiredColumnsError:
    prefixes = ("P02", "P03", "P05", "ANIO", "TRIM", "DOMINIO", "NUM_", "FACTOR", "OCUP")
    relevant = [column for column in available if column.upper().startswith(prefixes)]
    shown = relevant or list(available)
    preview = ", ".join(shown[:40])
    if len(shown) > 40:
        preview += ", ..."
    return RequiredColumnsError(
        f"El archivo '{path.name}' no contiene las columnas requeridas: "
        f"{', '.join(missing)}. Columnas disponibles relevantes: {preview}"
    )


def _safe_excel_cell(value: object, pd: Any) -> str | None:
    """Make a scalar safe for a StringType staging DataFrame."""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, Decimal):
        if not value.is_finite():
            return str(value)
        return format(value, "f")
    if isinstance(value, Real):
        numeric_value = float(value)
        if math.isnan(numeric_value):
            return None
        if math.isinf(numeric_value):
            return str(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return str(value)
    text = str(value).strip()
    return text or None


def _require_pyspark_functions() -> Any:
    try:
        from pyspark.sql import functions as F
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "Esta operación requiere PySpark 3.5.x en el entorno activo."
        ) from exc
    return F


def normalize_code(
    value: object,
    *,
    valid_codes: Sequence[object] | None = None,
    unknown: str = "DESCONOCIDO",
) -> str | None:
    """Normalize scalar survey codes such as ``1``, ``1.0`` and ``\"1\"``.

    Missing values remain ``None`` unless a set of valid codes is supplied, in
    which case missing/unrecognized values become ``unknown``.  Numeric zero is
    deliberately preserved (for example, educational code ``P03A03A = 0`` is
    not treated as missing).
    """

    missing = value is None
    if not missing:
        try:
            missing = bool(math.isnan(float(value)))
        except (TypeError, ValueError):
            missing = False
    if missing:
        return unknown if valid_codes is not None else None

    if isinstance(value, bool):
        normalised = "1" if value else "0"
    elif isinstance(value, Real):
        numeric = float(value)
        if math.isinf(numeric):
            normalised = str(value)
        elif numeric.is_integer():
            normalised = str(int(numeric))
        else:
            normalised = format(numeric, ".15g")
    else:
        normalised = str(value).strip()
        if not normalised:
            return unknown if valid_codes is not None else None
        # Decimal avoids turning an integral text code into an accidental float.
        try:
            decimal_value = Decimal(normalised)
            if decimal_value.is_finite() and decimal_value == decimal_value.to_integral_value():
                normalised = str(int(decimal_value))
        except (InvalidOperation, ValueError):
            pass

    if valid_codes is None:
        return normalised
    allowed = {normalize_code(code) for code in valid_codes}
    return normalised if normalised in allowed else unknown


def normalize_code_column(
    column: Any,
    *,
    valid_codes: Sequence[object] | None = None,
    unknown: str = "DESCONOCIDO",
) -> Any:
    """Spark-column counterpart to :func:`normalize_code`, without a Python UDF."""

    F = _require_pyspark_functions()
    text = F.trim(column.cast("string"))
    normalised = F.when(
        text.rlike(r"^[+-]?\d+\.0+$"), F.regexp_replace(text, r"\.0+$", "")
    ).otherwise(text)
    normalised = F.when(F.length(normalised) == 0, F.lit(None).cast("string")).otherwise(
        normalised
    )
    if valid_codes is None:
        return normalised
    allowed = [normalize_code(code) for code in valid_codes]
    return (
        F.when(normalised.isNull(), F.lit(unknown))
        .when(normalised.isin(*allowed), normalised)
        .otherwise(F.lit(unknown))
    )


def _numeric_column(column: Any) -> Any:
    """Safely coerce a Spark column to double, including under ANSI mode.

    A direct ``cast`` can throw on malformed survey text when
    ``spark.sql.ansi.enabled`` is true.  Guarding the cast keeps malformed
    values observable as null so the audit can report their exclusion.
    """

    F = _require_pyspark_functions()
    text = F.trim(column.cast("string"))
    ordinary_number = text.rlike(
        r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][+-]?\d+)?$"
    )
    lowered = F.lower(text)
    return (
        F.when(ordinary_number, text.cast("double"))
        .when(lowered == "nan", F.lit(float("nan")))
        .when(
            lowered.isin("infinity", "+infinity", "inf", "+inf"),
            F.lit(float("inf")),
        )
        .when(lowered.isin("-infinity", "-inf"), F.lit(float("-inf")))
        .otherwise(F.lit(None).cast("double"))
    )


def _finite_numeric(column: Any) -> Any:
    F = _require_pyspark_functions()
    numeric = _numeric_column(column)
    return (
        numeric.isNotNull()
        & (~F.isnan(numeric))
        & (numeric != F.lit(float("inf")))
        & (numeric != F.lit(float("-inf")))
    )


def _resolve_metadata(path: Path, metadata: Mapping[str, object] | None) -> dict[str, object]:
    supplied = dict(metadata or {})
    try:
        detected = detect_period_from_filename(path)
    except ValueError as detection_error:
        detected = {}
        if not {"periodo_archivo", "anio_archivo", "trimestre_calendario"}.issubset(supplied):
            raise ValueError(
                f"No se pudo derivar el período de '{path.name}'. Proporcione "
                "periodo_archivo, anio_archivo y trimestre_calendario explícitamente."
            ) from detection_error

    result: dict[str, object] = dict(detected)
    for key in ("periodo_archivo", "anio_archivo", "trimestre_calendario"):
        if key in supplied and key in detected and str(supplied[key]) != str(detected[key]):
            raise ValueError(
                f"Metadato {key!r}={supplied[key]!r} contradice el período detectado "
                f"en '{path.name}' ({detected[key]!r})."
            )
        if key in supplied:
            result[key] = supplied[key]

    period_match = re.fullmatch(
        r"\s*(20\d{2})\s*T\s*([1-4])\s*", str(result.get("periodo_archivo", ""))
    )
    if not period_match:
        raise ValueError(
            "periodo_archivo debe tener el formato YYYYTN; por ejemplo, '2026T1'."
        )
    period_year, period_quarter = map(int, period_match.groups())
    try:
        metadata_year = int(str(result["anio_archivo"]).strip())
        metadata_quarter = int(str(result["trimestre_calendario"]).strip())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "anio_archivo y trimestre_calendario deben ser enteros válidos."
        ) from exc
    if metadata_year != period_year or metadata_quarter != period_quarter:
        raise ValueError(
            "Los metadatos del período son inconsistentes: periodo_archivo debe "
            "coincidir con anio_archivo y trimestre_calendario."
        )
    return {
        "archivo_origen": path.name,
        "periodo_archivo": f"{period_year}T{period_quarter}",
        "anio_archivo": period_year,
        "trimestre_calendario": period_quarter,
    }


def _typed_source_expression(source: str, column: Any) -> Any:
    if source in _NUMERIC_SOURCE_COLUMNS:
        return _numeric_column(column)
    if source == "P05C16":
        return normalize_code_column(
            column, valid_codes=VALID_OCCUPATIONAL_CATEGORIES
        )
    if source in {"P03A03A", "DOMINIO"}:
        # Code 0 is preserved for education.  We do not have a closed list of
        # valid education/domain codes, so only blank or missing values are
        # replaced by the explicit category required for model pipelines.
        F = _require_pyspark_functions()
        return F.coalesce(normalize_code_column(column), F.lit("DESCONOCIDO"))
    if source in _CODE_SOURCE_COLUMNS:
        return normalize_code_column(column)
    return column


def load_excel_to_spark(
    path: str | Path,
    sheet_name: str | int = 0,
    spark: Any | None = None,
    selected_columns: Sequence[str] | Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
    *,
    required_columns: Sequence[str] | None = None,
    engine: str | None = "openpyxl",
) -> Any:
    """Load one Personas Excel workbook into the standard, narrow Spark schema.

    Only the requested fields are read with ``pandas.read_excel(usecols=...)``.
    The full header is inspected first solely to give a useful missing-column
    error.  ``TRIMESTRE`` is preserved as an original audit field; calendar
    quarter comes only from file provenance in ``trimestre_calendario``.
    """

    if spark is None:
        raise TypeError("load_excel_to_spark requiere una SparkSession en 'spark'.")
    excel_path = _as_existing_path(path)
    requested = _normalise_requested_columns(selected_columns)
    required = tuple(
        _clean_header(column)
        for column in (required_columns if required_columns is not None else requested)
    )
    missing_from_selection = [column for column in required if column not in requested]
    if missing_from_selection:
        raise ValueError(
            "Las columnas requeridas deben estar incluidas en selected_columns: "
            + ", ".join(missing_from_selection)
        )

    pd, raw_headers, clean_headers = _read_excel_header_info(
        excel_path, sheet_name, engine=engine
    )
    actual_by_clean = dict(zip(clean_headers, raw_headers))
    missing = [column for column in required if column not in actual_by_clean]
    if missing:
        raise _schema_error(excel_path, missing, clean_headers)
    selected_actual = [actual_by_clean[source] for source in requested]
    try:
        pandas_frame = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            usecols=selected_actual,
            dtype=object,
            engine=engine,
        )
    except Exception as exc:
        raise ValueError(
            f"No se pudo leer las columnas seleccionadas de '{excel_path.name}', "
            f"hoja {sheet_name!r}: {exc}"
        ) from exc

    # Stage every selected Excel value as text.  This avoids pandas/Spark type
    # inference failures for all-null columns; semantic casts happen in Spark.
    from pyspark.sql.types import StringType, StructField, StructType

    staged_sources = list(requested)
    pandas_frame.columns = [_clean_header(column) for column in pandas_frame.columns]
    records = [
        tuple(_safe_excel_cell(value, pd) for value in row)
        for row in pandas_frame[staged_sources].itertuples(index=False, name=None)
    ]
    staging_schema = StructType(
        [StructField(source, StringType(), nullable=True) for source in staged_sources]
    )
    staging = spark.createDataFrame(records, schema=staging_schema)

    F = _require_pyspark_functions()
    typed = staging.select(
        *[
            _typed_source_expression(source, F.col(source)).alias(target)
            for source, target in requested.items()
        ]
    )
    # The derived seniority is deliberately null until both source components
    # can be interpreted.  Audit filters explain any later exclusion.
    if {"antiguedad_anios", "antiguedad_meses"}.issubset(typed.columns):
        typed = typed.withColumn(
            "antiguedad",
            F.when(
                F.col("antiguedad_anios").isNotNull()
                & F.col("antiguedad_meses").isNotNull(),
                F.col("antiguedad_anios") + F.col("antiguedad_meses") / F.lit(12.0),
            ).otherwise(F.lit(None).cast("double")),
        )

    period_metadata = _resolve_metadata(excel_path, metadata)
    typed = (
        typed.withColumn("archivo_origen", F.lit(period_metadata["archivo_origen"]))
        .withColumn("periodo_archivo", F.lit(period_metadata["periodo_archivo"]))
        .withColumn("anio_archivo", F.lit(period_metadata["anio_archivo"]).cast("int"))
        .withColumn(
            "trimestre_calendario",
            F.lit(period_metadata["trimestre_calendario"]).cast("int"),
        )
    )
    return typed


def load_person_period(
    path: str | Path,
    spark: Any,
    *,
    sheet_name: str | int = 0,
    metadata: Mapping[str, object] | None = None,
) -> Any:
    """Convenience wrapper for loading the standard ENEIC Personas selection."""

    return load_excel_to_spark(
        path,
        sheet_name=sheet_name,
        spark=spark,
        selected_columns=SELECTED_SOURCE_COLUMNS,
        metadata=metadata,
        required_columns=REQUIRED_PERSON_COLUMNS,
    )


def _require_dataframe_columns(df: Any, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: faltan las columnas {', '.join(missing)}. "
            f"Disponibles: {', '.join(df.columns)}"
        )


def _missing_condition(column: Any) -> Any:
    F = _require_pyspark_functions()
    as_text = F.trim(column.cast("string"))
    return column.isNull() | F.isnan(_numeric_column(column)) | (F.length(as_text) == 0)


def missing_summary(
    df: Any,
    columns: Sequence[str] = MISSINGNESS_COLUMNS,
) -> Any:
    """Return Spark missingness counts and percentages, ordered by missingness.

    ``None``, numeric ``NaN`` and blank/whitespace-only strings are counted as
    missing.  The input is never converted to pandas.
    """

    requested = list(columns)
    if not requested:
        raise ValueError("columns no puede estar vacío.")
    _require_dataframe_columns(df, requested, "missing_summary")
    F = _require_pyspark_functions()
    total = df.count()
    # Evaluate one condition at a time.  A single wide aggregate repeats the
    # source-casting expression for every field and can exceed Spark's JVM
    # code-generation bytecode limit on the ENEIC schema.
    counts = {
        column: int(df.where(_missing_condition(F.col(column))).count())
        for column in requested
    }
    records = [
        (
            column,
            int(total),
            int(counts.get(column) or 0),
            (100.0 * int(counts.get(column) or 0) / total) if total else None,
        )
        for column in requested
    ]
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

    schema = StructType(
        [
            StructField("variable", StringType(), False),
            StructField("registros_totales", LongType(), False),
            StructField("faltantes", LongType(), False),
            StructField("porcentaje_faltante", DoubleType(), True),
        ]
    )
    return df.sparkSession.createDataFrame(records, schema=schema).orderBy(
        F.desc("faltantes"), F.asc("variable")
    )


def _filter_column_map(columns: Mapping[str, str] | None) -> dict[str, str]:
    defaults = {
        "edad": "edad",
        "ocupado": "ocupado",
        "categoria_ocupacional": "categoria_ocupacional",
        "salario_mensual": "salario_mensual",
        "antiguedad_anios": "antiguedad_anios",
        "antiguedad_meses": "antiguedad_meses",
        "antiguedad": "antiguedad",
        "horas_semanales": "horas_semanales",
    }
    if columns:
        defaults.update(columns)
    return defaults


def _filter_steps(df: Any, names: Mapping[str, str]) -> tuple[Any, list[tuple[int, str, Any]]]:
    """Build the mandated ordered filter conditions, adding seniority if needed."""

    F = _require_pyspark_functions()
    required_without_derived = [
        names[key]
        for key in (
            "edad",
            "ocupado",
            "categoria_ocupacional",
            "salario_mensual",
            "antiguedad_anios",
            "antiguedad_meses",
            "horas_semanales",
        )
    ]
    _require_dataframe_columns(df, required_without_derived, "apply_filters_with_audit")
    years = _numeric_column(F.col(names["antiguedad_anios"]))
    months = _numeric_column(F.col(names["antiguedad_meses"]))
    if names["antiguedad"] not in df.columns:
        df = df.withColumn(names["antiguedad"], years + months / F.lit(12.0))

    age = _numeric_column(F.col(names["edad"]))
    salary = _numeric_column(F.col(names["salario_mensual"]))
    seniority = _numeric_column(F.col(names["antiguedad"]))
    hours = _numeric_column(F.col(names["horas_semanales"]))
    occupied = normalize_code_column(F.col(names["ocupado"]))
    category = normalize_code_column(F.col(names["categoria_ocupacional"]))
    integer_months = F.abs(months - F.floor(months)) < F.lit(1e-9)

    # The expressions above are already doubles.  Do not send them through
    # _numeric_column a second time: repeating the coercion inside every audit
    # condition creates a very large Catalyst plan for the ENEIC union.
    def is_finite_double(value: Any) -> Any:
        return (
            value.isNotNull()
            & (~F.isnan(value))
            & (value != F.lit(float("inf")))
            & (value != F.lit(float("-inf")))
        )

    return df, [
        (1, "Edad interpretable (numérica y finita)", is_finite_double(age)),
        (2, "Edad mayor o igual a 15 años", age >= F.lit(15.0)),
        (3, "Persona ocupada (OCUPADOS == 1)", occupied == F.lit("1")),
        (
            4,
            "Categoría ocupacional válida (1, 2, 3 o 4)",
            category.isin(*VALID_OCCUPATIONAL_CATEGORIES),
        ),
        (5, "Salario interpretable (numérico)", salary.isNotNull()),
        (6, "Salario finito y mayor que cero", is_finite_double(salary) & (salary > 0)),
        (7, "Antigüedad en años válida (numérica y finita)", is_finite_double(years)),
        (
            8,
            "Antigüedad en meses entera entre 0 y 11",
            is_finite_double(months)
            & integer_months
            & (months >= 0)
            & (months <= 11),
        ),
        (9, "Antigüedad total mayor o igual a cero", is_finite_double(seniority) & (seniority >= 0)),
        (10, "Antigüedad total no mayor que la edad", seniority <= age),
        (11, "Horas semanales finitas y mayores que cero", is_finite_double(hours) & (hours > 0)),
        (12, "Horas semanales no mayores que 168", hours <= 168),
    ]


def apply_filters_with_audit(
    df: Any,
    columns: Mapping[str, str] | None = None,
) -> tuple[Any, Any]:
    """Apply the laboratory population filters and return ``(filtered, audit)``.

    The percentage in each audit row is calculated against the records entering
    that step.  No records are hidden with ``dropDuplicates`` or any implicit
    imputation.  Steps are fixed in the order required by the laboratory guide.
    """

    F = _require_pyspark_functions()
    names = _filter_column_map(columns)
    working, steps = _filter_steps(df, names)
    # Each audit step is an action.  A local checkpoint truncates the logical
    # plan between steps, rather than merely caching an ever-growing chain of
    # filters.  It needs no distributed checkpoint directory and is reliable
    # for this single Spark session; the caller persists the returned frame.
    working = working.localCheckpoint(eager=True)
    before = working.count()
    audit_rows: list[tuple[int, str, int, int, int, float | None]] = [
        (0, "Registros originales", before, 0, before, 0.0 if before else None)
    ]
    for step, description, condition in steps:
        after_frame = working.filter(condition).localCheckpoint(eager=True)
        after = after_frame.count()
        excluded = before - after
        percentage = (100.0 * excluded / before) if before else None
        audit_rows.append((step, description, before, excluded, after, percentage))
        working.unpersist()
        working, before = after_frame, after

    from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

    audit_schema = StructType(
        [
            StructField("paso", IntegerType(), False),
            StructField("descripcion", StringType(), False),
            StructField("registros_antes", LongType(), False),
            StructField("registros_excluidos", LongType(), False),
            StructField("registros_despues", LongType(), False),
            StructField("porcentaje_excluido", DoubleType(), True),
        ]
    )
    audit = working.sparkSession.createDataFrame(audit_rows, schema=audit_schema).orderBy(
        F.asc("paso")
    )
    return working, audit


# The brief uses both names; retaining an explicit alias avoids notebook drift.
filter_with_audit = apply_filters_with_audit


def descriptive_statistics(
    df: Any,
    columns: Sequence[str] = NUMERIC_ANALYSIS_COLUMNS,
    *,
    accuracy: int = 10_000,
) -> Any:
    """Compute requested descriptive statistics in Spark, one row per variable."""

    requested = list(columns)
    if not requested:
        raise ValueError("columns no puede estar vacío.")
    if accuracy <= 0:
        raise ValueError("accuracy debe ser un entero positivo.")
    _require_dataframe_columns(df, requested, "descriptive_statistics")
    F = _require_pyspark_functions()
    expressions: list[Any] = []
    for column in requested:
        value = F.when(_finite_numeric(F.col(column)), _numeric_column(F.col(column)))
        expressions.extend(
            [
                F.count(value).alias(f"{column}__n"),
                F.avg(value).alias(f"{column}__media"),
                F.percentile_approx(value, 0.5, accuracy).alias(f"{column}__mediana"),
                F.stddev_samp(value).alias(f"{column}__stddev"),
                F.min(value).alias(f"{column}__min"),
                F.percentile_approx(value, 0.25, accuracy).alias(f"{column}__p25"),
                F.percentile_approx(value, 0.75, accuracy).alias(f"{column}__p75"),
                F.percentile_approx(value, 0.95, accuracy).alias(f"{column}__p95"),
                F.max(value).alias(f"{column}__max"),
            ]
        )
    result = df.agg(*expressions).first().asDict()
    rows = [
        (
            position,
            column,
            int(result.get(f"{column}__n") or 0),
            result.get(f"{column}__media"),
            result.get(f"{column}__mediana"),
            result.get(f"{column}__stddev"),
            result.get(f"{column}__min"),
            result.get(f"{column}__p25"),
            result.get(f"{column}__p75"),
            result.get(f"{column}__p95"),
            result.get(f"{column}__max"),
        )
        for position, column in enumerate(requested)
    ]
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

    schema = StructType(
        [
            StructField("_orden", LongType(), False),
            StructField("variable", StringType(), False),
            StructField("n", LongType(), False),
            StructField("media", DoubleType(), True),
            StructField("mediana", DoubleType(), True),
            StructField("stddev", DoubleType(), True),
            StructField("min", DoubleType(), True),
            StructField("p25", DoubleType(), True),
            StructField("p75", DoubleType(), True),
            StructField("p95", DoubleType(), True),
            StructField("max", DoubleType(), True),
        ]
    )
    return df.sparkSession.createDataFrame(rows, schema=schema).orderBy("_orden").drop("_orden")


def median_by_group(
    df: Any,
    group_columns: str | Sequence[str],
    value_column: str = "salario_mensual",
    *,
    accuracy: int = 10_000,
    median_alias: str | None = None,
) -> Any:
    """Return Spark counts and approximate medians for one or more grouping fields."""

    groups = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    if not groups:
        raise ValueError("group_columns no puede estar vacío.")
    _require_dataframe_columns(df, [*groups, value_column], "median_by_group")
    if accuracy <= 0:
        raise ValueError("accuracy debe ser un entero positivo.")
    F = _require_pyspark_functions()
    value = _numeric_column(F.col(value_column))
    valid = _finite_numeric(value)
    alias = median_alias or (
        "mediana_salario" if value_column == "salario_mensual" else f"mediana_{value_column}"
    )
    return df.where(valid).groupBy(*groups).agg(
        F.count(value).alias("n"),
        F.percentile_approx(value, 0.5, accuracy).alias(alias),
    )


def prepare_kmeans_features(
    df: Any,
    feature_columns: Sequence[str] = KMEANS_FEATURE_COLUMNS,
    *,
    unscaled_col: str = "features_unscaled",
    features_col: str = "features",
    with_mean: bool = True,
    with_std: bool = True,
) -> tuple[Any, Any, Any]:
    """Assemble and standardize KMeans features with Spark ML.

    By default this uses exactly ``edad``, ``antiguedad`` and
    ``horas_semanales``--not salary.  It returns ``(prepared_df, assembler,
    scaler_model)`` so the notebook can reuse the fitted transformation.
    """

    features = list(feature_columns)
    if not features:
        raise ValueError("feature_columns no puede estar vacío.")
    if "salario_mensual" in features:
        raise ValueError(
            "salario_mensual no debe formar parte del vector KMeans inicial."
        )
    _require_dataframe_columns(df, features, "prepare_kmeans_features")
    try:
        from pyspark.ml.feature import StandardScaler, VectorAssembler
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("prepare_kmeans_features requiere pyspark.ml.") from exc
    F = _require_pyspark_functions()
    prepared = df
    for column in features:
        prepared = prepared.withColumn(column, _numeric_column(F.col(column)))
        prepared = prepared.filter(_finite_numeric(F.col(column)))
    assembler = VectorAssembler(
        inputCols=features,
        outputCol=unscaled_col,
        handleInvalid="error",
    )
    assembled = assembler.transform(prepared)
    scaler = StandardScaler(
        inputCol=unscaled_col,
        outputCol=features_col,
        withMean=with_mean,
        withStd=with_std,
    )
    scaler_model = scaler.fit(assembled)
    return scaler_model.transform(assembled), assembler, scaler_model


def evaluate_regression(
    predictions: Any,
    label_col: str = "salario_mensual",
    prediction_col: str = "prediction",
) -> dict[str, float]:
    """Evaluate a regression DataFrame with the three lab metrics in Spark.

    The caller is responsible for supplying a full validation or test
    DataFrame.  This helper deliberately performs no sampling and does not
    use pandas, so MAE, RMSE and R² always describe every prediction provided.
    """

    _require_dataframe_columns(
        predictions, [label_col, prediction_col], "evaluate_regression"
    )
    try:
        from pyspark.ml.evaluation import RegressionEvaluator
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("evaluate_regression requiere pyspark.ml.") from exc

    metrics: dict[str, float] = {}
    for output_name, metric_name in (("MAE", "mae"), ("RMSE", "rmse"), ("R2", "r2")):
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName=metric_name,
        )
        metrics[output_name] = float(evaluator.evaluate(predictions))
    return metrics


def make_supervised_pipeline(estimator: Any) -> Any:
    """Create the leakage-safe feature pipeline mandated by the laboratory.

    The returned, *unfitted* pipeline has exactly the three numeric and three
    categorical predictors specified in the brief.  Calling ``fit(train)``
    therefore learns StringIndexer and OneHotEncoder stages only from the
    training data supplied by the caller.
    """

    try:
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("make_supervised_pipeline requiere pyspark.ml.") from exc

    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}__index",
            handleInvalid="keep",
        )
        for column in SUPERVISED_CATEGORICAL_PREDICTORS
    ]
    encoded_columns = [f"{column}__ohe" for column in SUPERVISED_CATEGORICAL_PREDICTORS]
    encoder = OneHotEncoder(
        inputCols=[f"{column}__index" for column in SUPERVISED_CATEGORICAL_PREDICTORS],
        outputCols=encoded_columns,
        handleInvalid="keep",
        dropLast=True,
    )
    assembler = VectorAssembler(
        inputCols=[*SUPERVISED_NUMERIC_PREDICTORS, *encoded_columns],
        outputCol="features",
        handleInvalid="error",
    )
    return Pipeline(stages=[*indexers, encoder, assembler, estimator])


def build_linear_regression_pipeline(
    *,
    reg_param: float,
    elastic_net_param: float,
    max_iter: int = 100,
    label_col: str = "salario_mensual",
    prediction_col: str = "prediction",
) -> Any:
    """Return a fresh regularized Spark LinearRegression pipeline."""

    try:
        from pyspark.ml.regression import LinearRegression
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("build_linear_regression_pipeline requiere pyspark.ml.") from exc
    estimator = LinearRegression(
        featuresCol="features",
        labelCol=label_col,
        predictionCol=prediction_col,
        regParam=float(reg_param),
        elasticNetParam=float(elastic_net_param),
        maxIter=int(max_iter),
        standardization=True,
    )
    return make_supervised_pipeline(estimator)


def build_random_forest_pipeline(
    *,
    num_trees: int,
    max_depth: int,
    seed: int = SEED,
    max_bins: int = 16,
    max_memory_in_mb: int = 64,
    label_col: str = "salario_mensual",
    prediction_col: str = "prediction",
) -> Any:
    """Return a fresh, bounded-memory Spark RF pipeline without scaling.

    The ENEIC notebook runs locally on Windows.  Keeping histogram bins and
    per-task RF memory bounded avoids exhausting the JVM when multiple local
    workers build tree histograms concurrently; it does not alter the six
    required predictors or use any test data during fitting.
    """

    try:
        from pyspark.ml.regression import RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("build_random_forest_pipeline requiere pyspark.ml.") from exc
    estimator = RandomForestRegressor(
        featuresCol="features",
        labelCol=label_col,
        predictionCol=prediction_col,
        numTrees=int(num_trees),
        maxDepth=int(max_depth),
        seed=int(seed),
        maxBins=int(max_bins),
        maxMemoryInMB=int(max_memory_in_mb),
    )
    return make_supervised_pipeline(estimator)


def with_residual(
    predictions: Any,
    *,
    label_col: str = "salario_mensual",
    prediction_col: str = "prediction",
    residual_col: str = "residuo",
) -> Any:
    """Add ``real - prediction`` so positive values mean underprediction."""

    _require_dataframe_columns(predictions, [label_col, prediction_col], "with_residual")
    F = _require_pyspark_functions()
    return predictions.withColumn(
        residual_col,
        _numeric_column(F.col(label_col)) - _numeric_column(F.col(prediction_col)),
    )


def grouped_error_metrics(
    predictions: Any,
    group_col: str,
    *,
    label_col: str = "salario_mensual",
    prediction_col: str = "prediction",
    residual_col: str = "residuo",
) -> Any:
    """Calculate count, MAE and mean signed error by one categorical group."""

    _require_dataframe_columns(
        predictions, [group_col, label_col, prediction_col], "grouped_error_metrics"
    )
    F = _require_pyspark_functions()
    with_error = (
        predictions
        if residual_col in predictions.columns
        else with_residual(
            predictions,
            label_col=label_col,
            prediction_col=prediction_col,
            residual_col=residual_col,
        )
    )
    group = F.coalesce(F.trim(F.col(group_col).cast("string")), F.lit("DESCONOCIDO"))
    return (
        with_error.withColumn("_grupo", group)
        .groupBy("_grupo")
        .agg(
            F.count("*").alias("n"),
            F.avg(F.abs(F.col(residual_col))).alias("MAE"),
            F.avg(F.col(residual_col)).alias("error_medio"),
        )
        .withColumnRenamed("_grupo", group_col)
        .orderBy(F.desc("n"), F.asc(group_col))
    )


def deterministic_sample(
    df: Any,
    *,
    max_rows: int = 5_000,
    seed: int = SEED,
    key_columns: Sequence[str] = ("periodo_archivo", "NUM_HOGAR", "NUM_PERSONA"),
) -> Any:
    """Select one deterministic Spark sample, retaining it for multiple models.

    The requested ENEIC identity is hashed with the seed.  The full result is
    still evaluated elsewhere; this function is only for bounded graphics.
    """

    if max_rows <= 0:
        raise ValueError("max_rows debe ser positivo.")
    _require_dataframe_columns(df, list(key_columns), "deterministic_sample")
    F = _require_pyspark_functions()
    serialised = [F.lit(str(seed))] + [
        F.coalesce(F.col(column).cast("string"), F.lit("<NULO>"))
        for column in key_columns
    ]
    sample_hash = F.sha2(F.concat_ws("¦", *serialised), 256)
    return (
        df.withColumn("__sample_hash", sample_hash)
        .orderBy(F.col("__sample_hash"), *[F.col(column).cast("string") for column in key_columns])
        .limit(int(max_rows))
        .drop("__sample_hash")
    )


def write_small_table(frame: Any, destination: str | Path, *, max_rows: int = 10_000) -> Path:
    """Write a small aggregate to one CSV after enforcing a row-count ceiling."""

    if max_rows <= 0:
        raise ValueError("max_rows debe ser positivo.")
    count = frame.limit(int(max_rows) + 1).count()
    if count > max_rows:
        raise ValueError(
            f"La tabla tiene mÃ¡s de {max_rows:,} filas; no debe convertirse a pandas."
        )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    frame.toPandas().to_csv(destination_path, index=False, encoding="utf-8-sig")
    return destination_path


def feature_importance_table(
    random_forest_model: Any,
    transformed_frame: Any,
    *,
    features_col: str = "features",
) -> Any:
    """Return a small pandas table of RF importances using Spark metadata when present.

    When Spark does not expose an encoded attribute name, the table labels only
    its vector position as ``feature_<index>``; it does not guess a category.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("feature_importance_table requiere pandas.") from exc
    if features_col not in transformed_frame.columns:
        raise ValueError(f"No existe la columna de features: {features_col}")
    metadata = transformed_frame.schema[features_col].metadata.get("ml_attr", {})
    attributes = metadata.get("attrs", {})
    names: dict[int, str] = {}
    for attribute_group in attributes.values():
        for attribute in attribute_group:
            index = attribute.get("idx")
            name = attribute.get("name")
            if index is not None and name:
                names[int(index)] = str(name)
    importances = list(random_forest_model.featureImportances)
    rows = [
        {
            "indice": index,
            "feature": names.get(index, f"feature_{index}"),
            "importancia": float(importance),
        }
        for index, importance in enumerate(importances)
    ]
    return pd.DataFrame(rows).sort_values("importancia", ascending=False, ignore_index=True)


__all__ = [
    "EXCEL_SUFFIXES",
    "KMEANS_FEATURE_COLUMNS",
    "MISSINGNESS_COLUMNS",
    "NUMERIC_ANALYSIS_COLUMNS",
    "PERIOD_METADATA_COLUMNS",
    "REQUIRED_PERSON_COLUMNS",
    "REQUIRED_COLUMNS",
    "SEED",
    "SELECTED_SOURCE_COLUMNS",
    "SELECTED_COLUMNS",
    "STANDARD_ANALYTIC_COLUMNS",
    "STANDARD_COLUMN_MAP",
    "SUPERVISED_CATEGORICAL_PREDICTORS",
    "SUPERVISED_NUMERIC_PREDICTORS",
    "SUPERVISED_PREDICTORS",
    "VALID_OCCUPATIONAL_CATEGORIES",
    "RequiredColumnsError",
    "apply_filters_with_audit",
    "build_linear_regression_pipeline",
    "build_random_forest_pipeline",
    "deterministic_sample",
    "descriptive_statistics",
    "detect_period_from_filename",
    "evaluate_regression",
    "feature_importance_table",
    "filter_with_audit",
    "find_person_files",
    "grouped_error_metrics",
    "load_excel_to_spark",
    "load_person_period",
    "make_supervised_pipeline",
    "median_by_group",
    "missing_summary",
    "normalize_code",
    "normalize_code_column",
    "prepare_kmeans_features",
    "read_excel_columns",
    "with_residual",
    "write_small_table",
]
