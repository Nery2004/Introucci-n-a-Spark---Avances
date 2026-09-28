"""Static validation for the Laboratorio 7 deliverable.

This script deliberately checks the notebook source rather than assuming that
saved outputs imply compliance. Run it after generating the notebook.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Laboratorio_7_SparkML_Nery_Molina_23218.ipynb"


def required_source(source: str, *terms: str) -> bool:
    return all(term in source for term in terms)


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def metrics_are_finite(path: Path) -> bool:
    """Check that CSV metrics contain real finite values, not placeholders."""
    if not nonempty_file(path):
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    for row in rows:
        for field in ("MAE", "RMSE", "R2"):
            value = row.get(field)
            if value is None:
                continue
            try:
                if not math.isfinite(float(value)):
                    return False
            except ValueError:
                return False
    return True


def main() -> int:
    if not NOTEBOOK.is_file():
        print(f"FALLÓ: no existe el notebook esperado: {NOTEBOOK}")
        return 1

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    lower_source = source.lower()
    utils_source = (ROOT / "src" / "utils.py").read_text(encoding="utf-8")

    checks = {
        "1. Notebook existe y tiene JSON válido": True,
        "2. No usa sklearn": "sklearn" not in lower_source,
        "3. Usa pyspark.ml": "pyspark.ml" in lower_source,
        "4. Usa unionByName": "unionByName" in source,
        "5. Train/validation procede solo de 2025": required_source(
            source, "train_2025, validation_2025", "df_2025_prepared.randomSplit"
        ) and "df_2026_prepared.randomSplit" not in source,
        "6. P05D01 es el objetivo": '"P05D01"' in source,
        "7. Están exactamente los seis predictores supervisados": required_source(
            source, "SUPERVISED_PREDICTORS"
        ) and required_source(
            utils_source,
            "SUPERVISED_NUMERIC_PREDICTORS",
            '"edad"',
            '"antiguedad"',
            '"horas_semanales"',
            "SUPERVISED_CATEGORICAL_PREDICTORS",
            '"nivel_educativo"',
            '"categoria_ocupacional"',
            '"dominio"',
        ),
        "8. Usa VectorAssembler y Correlation.corr": required_source(
            source, "VectorAssembler", "Correlation.corr"
        ),
        "9. KMeans utiliza Spark": required_source(source, "KMeans(", "ClusteringEvaluator"),
        "10. KMeans prueba K=2,3,4,5": "for k in (2, 3, 4, 5)" in source,
        "11. Linear Regression prueba al menos dos configuraciones": all(
            name in source for name in ("LR_1", "LR_2")
        ),
        "12. Random Forest prueba al menos dos configuraciones": all(
            name in source for name in ("RF_1", "RF_2")
        ),
        "13. Reporta MAE": '"MAE"' in source,
        "14. Reporta RMSE": '"RMSE"' in source,
        "15. Reporta R2": '"R2"' in source,
        "16. Hay evaluación final de 2026": required_source(source, "pred_lr_2026", "pred_rf_2026"),
        "17. Incluye gráficos real vs. predicho": "real_vs_predicho" in source,
        "18. Incluye gráficos de residuos": "residuos_vs_predicho" in source,
        "19. Incluye métricas por nivel educativo": "errors_by_education" in source,
        "20. Incluye métricas por dominio": "errors_by_domain" in source,
        "21. Incluye análisis por bandas salariales": "errors_by_salary_band" in source,
        "22. Compara validation contra test y calcula deltas": required_source(
            source, "validation_vs_test_rows", "delta_MAE", "delta_RMSE"
        ),
        "23. Incluye discusión final": "# Discusión final" in source,
        "24. Incluye conclusiones": "# Conclusiones" in source,
        "25. README existe": (ROOT / "README.md").is_file(),
    }

    if "--source-only" not in sys.argv:
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        errors = [
            output
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        tables = ROOT / "outputs" / "tables"
        figures = ROOT / "outputs" / "figures"
        expected_tables = (
            "descriptive_stats.csv",
            "missing_summary.csv",
            "filter_audit.csv",
            "correlation_matrix.csv",
            "kmeans_scores.csv",
            "cluster_profiles.csv",
            "lr_validation_metrics.csv",
            "rf_validation_metrics.csv",
            "model_comparison_validation.csv",
            "model_comparison_test_2026.csv",
            "validation_vs_test.csv",
            "errors_by_education.csv",
            "errors_by_domain.csv",
            "errors_by_salary_band.csv",
        )
        checks.update({
            "26. Todas las celdas de código fueron ejecutadas": bool(code_cells) and all(
                cell.execution_count is not None for cell in code_cells
            ),
            "27. El notebook no contiene salidas de error": not errors,
            "28. Las tablas requeridas existen y no están vacías": all(
                nonempty_file(tables / name) for name in expected_tables
            ),
            "29. Las figuras requeridas existen y no están vacías": all(
                nonempty_file(figures / name)
                for name in (
                    "correlation_heatmap_2025.png",
                    "kmeans_silhouette_vs_k_2025.png",
                    "real_vs_predicho_linear_regression_2026.png",
                    "real_vs_predicho_random_forest_2026.png",
                    "residuos_vs_predicho_linear_regression_2026.png",
                    "residuos_vs_predicho_random_forest_2026.png",
                    "mae_por_banda_salario_2026.png",
                )
            ),
            "30. Los modelos finales existen": all(
                (ROOT / "models" / name).is_dir()
                for name in ("linear_regression_final", "random_forest_final")
            ),
            "31. Las métricas de validación y test son finitas": all(
                metrics_are_finite(tables / name)
                for name in (
                    "lr_validation_metrics.csv",
                    "rf_validation_metrics.csv",
                    "model_comparison_validation.csv",
                    "model_comparison_test_2026.csv",
                )
            ),
        })

    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{'OK' if passed else 'FALLÓ'}: {name}")
    if failed:
        print(f"\nValidación incompleta: {len(failed)} verificación(es) fallaron.")
        return 1
    print("\nValidación estática completada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
