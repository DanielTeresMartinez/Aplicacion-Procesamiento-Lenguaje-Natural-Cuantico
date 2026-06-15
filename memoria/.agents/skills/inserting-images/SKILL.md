---
name: inserting-images
description: Automate the insertion, conversion, and formatting of images for a LaTeX proyect. Use when the user asks to add a new image into the proyect.
---

# Inserting images

## Purpose
The goal is to ensure all images follow a professional academic standard (using LaTeX), have correct dimensions, and include proper source citations.

## Operational Workflow
1. **Format Validation & Conversion**: 
   - Check if the image in the `imagenes/` folder is `.png` or `.jpg`.
   - If it is any other format (e.g., .webp, .svg, .pdf, .bmp), the agent MUST use the image processing tool `convert` to convert it to `.png` before generating the code. If a new image with the same name but the correct extension (`.png`) was created sucessfully, then use the tool `rm` to remove the image with the wrong format.

2. **Dimension Analysis**:
   - Use the available tool to check image resolution.
   - Adjust `width=X.X\linewidth` based on the aspect ratio and complexity. e.g:
     - Large/Complex diagrams: 0.8\linewidth.
     - Standard images: 0.5\linewidth.
     - Small icons or simple shapes: 0.4\linewidth.

3. **Contextual Captioning**:
   - Analyze the surrounding text of section topic to generate a descriptive, academic caption.
   - **Citations**: If a `\cite` key or a URL is provided, append it to the caption (e.g., "Source: Elaborated from [Source]").

4. **Labeling Strategy**:
   - Automatically generate a unique and descriptive `\label{fig:keyword}` based on the image content.

## LaTeX Template Output
The agent must output the code in this **exact format:**

\begin{figure}[h]
\centering
\includegraphics[width=X.X\linewidth]{imagenes/filename.extension}
\caption{Descriptive text based on context. Source: [Citation/Link].}
\label{fig:descriptive_label}
\end{figure}

## Required Capabilities
- Image Metadata Inspection (size, format).
- Image Format Conversion.
- Contextual Document Reading (current LaTeX section).