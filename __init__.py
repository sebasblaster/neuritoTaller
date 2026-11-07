"""Entradas públicas de Neurito."""

import runpy


def main() -> None:
    """Ejecuta la interfaz original de la mascota."""
    runpy.run_module("neurito.main", run_name="__main__")
