"""Typed errors raised by the proteinviz core.

Front-ends (Streamlit app, CLI, notebooks) turn these into user-facing messages.
The core never substitutes made-up data when something goes wrong.
"""


class ProteinVizError(Exception):
    """Base class for all proteinviz errors."""


class InvalidIdentifier(ProteinVizError):
    """The identifier is not a recognisable UniProt accession, gene name or PDB ID."""


class NotFound(ProteinVizError):
    """The source answered, but has no record for the requested identifier."""


class SourceUnavailable(ProteinVizError):
    """The remote source could not be reached or returned an error."""


class OfflineCacheMiss(SourceUnavailable):
    """Offline mode is on and the request is not in the HTTP cache."""
