Karachi PM2.5 paper — Overleaf upload package
==============================================

Files in this folder
--------------------
paper.tex         The LaTeX source. Uses \documentclass{iopjournal}.
figs/             16 figure PNGs referenced by paper.tex.
README.txt        This file.

Steps to compile in Overleaf
----------------------------
1. Go to https://www.overleaf.com and start a new project (blank).
2. In the Overleaf file panel (top-left), click the upload icon
   (or drag-and-drop the entire `overleaf_package` folder) and upload:
     - paper.tex
     - the entire figs/ folder
   You can also drag the whole `overleaf_package/` folder at once —
   Overleaf will preserve the folder structure.
3. Open paper.tex in the Overleaf editor.
4. Click "Recompile" (top-right). You will see an error like:

       ! LaTeX Error: File `iopjournal.cls' not found.

   This is expected: iopjournal.cls is a proprietary IOPP class that
   is not bundled with vanilla LaTeX. To obtain it:

   a. If you have an IOPP submission account, log in and download
      the LaTeX template from the journal's author guidelines page.
      Place iopjournal.cls at the project root (next to paper.tex).
   b. Alternatively, if you just want to see the paper rendered
      before submission, change the first line of paper.tex from

          \documentclass{iopjournal}

      to

          \documentclass[11pt,a4paper]{article}
          \usepackage{graphicx}
          \usepackage{amsmath,amssymb}
          \usepackage{hyperref}

      and remove the `\articletype{}` line near the top. The IOPP
      commands (\ack, \funding, \roles, \data, \suppdata) will need
      to be replaced with manual \section*{} blocks. This is a 5-
      minute find-and-replace; let me know if you want me to do it.
5. Click Recompile again. The PDF should appear in the right pane.
6. To download the PDF: click the "Download PDF" button at the top
   of the preview pane. Save it anywhere on your computer; double-
   click to open.

Notes
-----
* The figures already match the path convention used in paper.tex
  (`figs/figXX_name.png`). No path edits needed.
* The bibliography is hand-formatted using a thebibliography
  environment — no external .bib file is required.
* If Overleaf complains about missing packages, click "Settings →
  Compiler" and ensure the compiler is "pdfLaTeX" (default).
* Estimated compile time: ~10 seconds. Output: ~22-page PDF.

Contact
-------
Sidhart Sami
sidhart.samir.punjabi@gmail.com
ORCID: 0009-0003-8133-1230