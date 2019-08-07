"""Module to remove the noise from Project Gutenberg texts."""


from __future__ import absolute_import, unicode_literals
from builtins import str
from itertools import chain
import os

# from gutenberg._domain_model.text import TEXT_END_MARKERS
# from gutenberg._domain_model.text import TEXT_START_MARKERS
# from gutenberg._domain_model.text import LEGALESE_END_MARKERS
# from gutenberg._domain_model.text import LEGALESE_START_MARKERS

# -*- coding: utf-8 -*-
"""Data module that contains strings that mark the start and end of a Project
Gutenberg disclaimer/header."""


TEXT_START_MARKERS = tuple((
    "*END*THE SMALL PRINT",
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "This etext was prepared by",
    "E-text prepared by",
    "Produced by",
    "Distributed Proofreading Team",
    "Proofreading Team at http://www.pgdp.net",
    "http://gallica.bnf.fr)",
    "      http://archive.org/details/",
    "http://www.pgdp.net",
    "by The Internet Archive)",
    "by The Internet Archive/Canadian Libraries",
    "by The Internet Archive/American Libraries",
    "public domain material from the Internet Archive",
    "Internet Archive)",
    "Internet Archive/Canadian Libraries",
    "Internet Archive/American Libraries",
    "material from the Google Print project",
    "*END THE SMALL PRINT",
    "***START OF THE PROJECT GUTENBERG",
    "This etext was produced by",
    "*** START OF THE COPYRIGHTED",
    "The Project Gutenberg",
    "http://gutenberg.spiegel.de/ erreichbar.",
    "Project Runeberg publishes",
    "Beginning of this Project Gutenberg",
    "Project Gutenberg Online Distributed",
    "Gutenberg Online Distributed",
    "the Project Gutenberg Online Distributed",
    "Project Gutenberg TEI",
    "This eBook was prepared by",
    "http://gutenberg2000.de erreichbar.",
    "This Etext was prepared by",
    "This Project Gutenberg Etext was prepared by",
    "Gutenberg Distributed Proofreaders",
    "Project Gutenberg Distributed Proofreaders",
    "the Project Gutenberg Online Distributed Proofreading Team",
    "**The Project Gutenberg",
    "*SMALL PRINT!",
    "More information about this book is at the top of this file.",
    "tells you about restrictions in how the file may be used.",
    "l'authorization à les utilizer pour preparer ce texte.",
    "of the etext through OCR.",
    "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
    "We need your donations more than ever!",
    " *** START OF THIS PROJECT GUTENBERG",
    "****     SMALL PRINT!",
    '["Small Print" V.',
    '      (http://www.ibiblio.org/gutenberg/',
    'and the Project Gutenberg Online Distributed Proofreading Team',
    'Mary Meehan, and the Project Gutenberg Online Distributed Proofreading',
    '                this Project Gutenberg edition.',
))


TEXT_END_MARKERS = tuple((
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of The Project Gutenberg",
    "Ende dieses Project Gutenberg",
    "by Project Gutenberg",
    "End of Project Gutenberg",
    "End of this Project Gutenberg",
    "Ende dieses Projekt Gutenberg",
    "        ***END OF THE PROJECT GUTENBERG",
    "*** END OF THE COPYRIGHTED",
    "End of this is COPYRIGHTED",
    "Ende dieses Etextes ",
    "Ende dieses Project Gutenber",
    "Ende diese Project Gutenberg",
    "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
    "Fin de Project Gutenberg",
    "The Project Gutenberg Etext of ",
    "Ce document fut presente en lecture",
    "Ce document fut présenté en lecture",
    "More information about this book is at the top of this file.",
    "We need your donations more than ever!",
    "END OF PROJECT GUTENBERG",
    " End of the Project Gutenberg",
    " *** END OF THIS PROJECT GUTENBERG",
))


LEGALESE_START_MARKERS = tuple(("<<THIS ELECTRONIC VERSION OF",))


LEGALESE_END_MARKERS = tuple(("SERVICE THAT CHARGES FOR DOWNLOAD",))


def filter_headers(lines, sep=str(os.linesep)):
    out = []
    i = 0
    ignore_section = False

    for line in lines:

        # If it's the end of the header, delete the output produced so far.
        # May be done several times, if multiple lines occur indicating the
        # end of the header
        if i <= 600 and line.startswith(TEXT_START_MARKERS):
            out = []
            continue

        # Check if reached footer
        if i >= 100 and line.startswith(TEXT_END_MARKERS):
            break

        if line.startswith(LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif line.startswith(LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1
    
    return out

def strip_headers(text, input_sep=str(os.linesep), output_sep=str(os.linesep)):
    """Remove lines that are part of the Project Gutenberg header or footer.
    Note: this function is a port of the C++ utility by Johannes Krugel. The
    original version of the code can be found at:
    http://www14.in.tum.de/spp1307/src/strip_headers.cpp

    Args:
        text (unicode): The body of the text to clean up.

    Returns:
        unicode: The text with any non-text content removed.

    """
    lines = text.splitlines()
    
    lines = filter_headers(lines, input_sep)

    return output_sep.join(lines)


def _main():
    """Command line interface to the module.

    """
    from argparse import ArgumentParser, FileType
    from gutenberg import Error
    from gutenberg._util.os import reopen_encoded

    parser = ArgumentParser(description='Remove headers and footers from a '
                                        'Project Gutenberg text')
    parser.add_argument('infile', type=FileType('r'))
    parser.add_argument('outfile', type=FileType('w'))
    args = parser.parse_args()

    try:
        with reopen_encoded(args.infile, 'r', 'utf8') as infile:
            text = infile.read()
            clean_text = strip_headers(text)

        with reopen_encoded(args.outfile, 'w', 'utf8') as outfile:
            outfile.write(clean_text)
    except Error as error:
        parser.error(str(error))


if __name__ == '__main__':
    _main()
