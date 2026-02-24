# Project Structure: Bachelor's Thesis (TFG)

This document describes the structure and organization of the project for the Bachelor's Thesis (TFG) titled "Application of Quantum Natural Language Processing" ("Aplicación Del Procesamiento Del Lenguaje Natural Cuántico").

The agent must use this document to orient itself within the project and know where to find or add content.

## Technology and Language
- **Main format**: LaTeX (`.tex`).
- **Encoding**: UTF-8.
- **Document language**: Spanish (`babel` configured in Spanish).
- **Bibliographic package**: `biblatex` using the `biber` backend.

## File and Directory Structure

The project is divided in a modular way, allowing simple and organized handling through the use of the `\input{}` directive or other inclusion commands.

*   `proyecto.tex`: This is the **main or root** file of the project. It loads all necessary packages (`hyperref`, `biblatex`, mathematical packages, etc.), defines the metadata (title, author), and coordinates the structural inclusion of the rest of the document (covers, chapters, glossaries, and bibliography).
*   `capitulos/`: Contains individual files for each chapter of the thesis. For example:
    *   `01_Introduccion.tex`
    *   `02_NumerosComplejos.tex`
    *   `03_NotacionDirac.tex`
    *   *And subsequent chapters of the document.*
    *   **Note:** When making modifications to the content of the thesis, the agent must always target the corresponding chapter within this directory, rather than modifying the main file.
*   `bibliografia/`: Directory containing the `bibliografia.bib` file, which is used to store all references in BibLaTeX format and to be able to cite them later using commands like `\cite{}` or similar, depending on the location.
*   `glosario/`: Stores the configuration and list of acronyms and definitions (e.g., `entradasGlosario.tex`).
*   `portada/`: File intended to build the cover and main page of the document (`portada.tex`).
*   `prefacios/`: Contains sections prior to the table of contents, such as abstract, acknowledgments, or previous introductions (`prefacio.tex`).
*   `imagenes/`: General directory intended to store all images referenced in the documentation.
*   `.agent/`: Space dedicated to configurations and aids for the autonomous agent (e.g., *skills*), where this base structure file is also located.

## Additional Considerations
- To insert text, citations, or other elements into the body of the document, do not modify `proyecto.tex`; instead, find the specific individual resource, such as a particular chapter in `capitulos/`.
- All bibliographic references must be rigorously formatted in BibLaTeX style in `bibliografia/bibliografia.bib`, adapting the fields depending on the type (article, website, post, etc.).
