---
name: reference-insertor
description: Use this skill when the user asks to add a reference and cite it in some place of the file.
---

# Reference Insertor

## Goal
Add a reference in the bibliography and cite it in the specific place to reduce the work.

## Instructions
- **MANDATORY FIRST STEP:** Before starting any process, you must read the project's structure file located in `../references/proyectDoc.md` so that you know the structure of the project, directories, files used (like `proyecto.tex`), languages and framework.
- Read the references data that the user wants to insert. This could come in 1 of 3 possible ways:
  1) **Scientific paper format:** The user gives the data of the cite of an article.
  2) **Web page format:** The user gives the link of a webpage. In that case, you **must** use the tools to make web scrapping to access that web and take the most relevant information.
  3) **Initial data of scientific paper format:** In case the user does not send the reference data following *step 1* format, it will gives the title, author, publication date, etc. of the paper. That's mean, the first page of the paper and you must infeer the data needed for the refrence.
- Extract the following data to add the new entry (or entries) following the specific package that is being used. The steps to extract the data needed for each entry are the followings:
  **COMMON RULES FOR ANY REFERENCE:**
  1. Generate a unique ID (key) by combining the surname/organization, a keyword from the title, and the year (e.g., `ferrovial_complex_numbers_2026`). Everything in lowercase.
  2. Extract the `title`.
  3. Extract the `author`. If there are multiple authors, separate them with " and ". If it is an institution or corpuste company (without a physical author), put it in double curly braces (e.g., `author = {{IBM}}`).
  4. Extract the original publication date. Use `date = {YYYY-MM-DD}` if you have the exact date, or `year = {YYYY}` if you only know the year.

  **CONDITIONALS BASED ON THE SOURCE TYPE:**

  -> **If it is a WEBPAGE, News, or Post (use `@online`):**
  - It is mandatory to extract the `url`.
  - Always add the `urldate = {YYYY-MM-DD}` field with today's date.
  - Look for the name of the website or company and add it in `organization`.

  -> **If it is a PAPER or SCIENTIFIC ARTICLE (use `@article`):**
  - Extract the name of the journal in `journal`.
  - Extract the volume in `volume`, the issue number in `number`, and the identifier in `doi`.
  - Extract the page range in `pages`. Extract the URL to the PDF if it is public.

  -> **If it is an INSTITUTIONAL REPORT or STUDY NOTES (use `@report`):**
  - Extract the `institution` backing the text and the `type` (e.g., "Technical Report", "Study material"). Always ensure the inclusion of the `url` and `urldate`.
- Add the entry on the specific file using the tools for reading and writing in a file.
- After adding the entry or new entries, use the syntax of the package to reference it where the user told to do it. The user will always tell to do it in the files of chapters.

## Constraints
- If you can not extract some information of the step *Conditionals Based On The Source type* continue with the data that you have, never stops.
- If some data of common rules is missing, don't add that entry and tell the user to give more information.