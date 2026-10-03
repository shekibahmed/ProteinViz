"""Typed records returned by the proteinviz core.

Every record carries a :class:`Provenance` so any number shown to a user can be
traced back to the database, identifier and retrieval time it came from.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source: str = Field(description="Database name, e.g. 'UniProt', 'STRING', 'RCSB PDB'.")
    url: str = Field(description="Human-browsable URL of the record.")
    version: str | None = Field(default=None, description="Database release, if known.")
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def at(cls, source: str, url: str, version: str | None = None, fetched_at: float | None = None):
        ts = datetime.fromtimestamp(fetched_at, UTC) if fetched_at else datetime.now(UTC)
        return cls(source=source, url=url, version=version, retrieved_at=ts)


class Domain(BaseModel):
    description: str
    start: int
    end: int


class ProteinRecord(BaseModel):
    accession: str
    entry_name: str | None = None
    protein_name: str | None = None
    gene_names: list[str] = []
    organism: str | None = None
    taxon_id: int | None = None
    reviewed: bool = False
    sequence: str
    length: int
    function: str | None = None
    subcellular_locations: list[str] = []
    keywords: list[str] = []
    domains: list[Domain] = []
    pdb_ids: list[str] = []
    provenance: Provenance

    @property
    def primary_gene(self) -> str | None:
        return self.gene_names[0] if self.gene_names else None


class SequenceProperties(BaseModel):
    length: int
    molecular_weight_da: float
    isoelectric_point: float
    gravy: float
    aromaticity: float
    instability_index: float
    method: str = "Biopython ProtParam (average isotopic masses; Bjellqvist pI)"


class StringInteraction(BaseModel):
    """One STRING edge. Scores are STRING's 0–1 confidence per evidence channel."""

    string_id_a: str
    string_id_b: str
    name_a: str
    name_b: str
    score: float
    experimental: float = 0.0
    database: float = 0.0
    textmining: float = 0.0
    coexpression: float = 0.0
    neighborhood: float = 0.0
    fusion: float = 0.0
    cooccurrence: float = 0.0


class StringNetwork(BaseModel):
    query: str
    species: int
    network_type: str
    required_score: int
    interactions: list[StringInteraction]
    provenance: Provenance

    @property
    def nodes(self) -> list[str]:
        names = {i.name_a for i in self.interactions} | {i.name_b for i in self.interactions}
        return sorted(names)


class PolymerEntity(BaseModel):
    entity_id: str
    description: str | None = None
    chains: list[str] = []
    uniprot_accessions: list[str] = []
    organism: str | None = None


class PdbEntry(BaseModel):
    pdb_id: str
    title: str | None = None
    method: str | None = None
    resolution: float | None = None
    release_date: str | None = None
    entities: list[PolymerEntity] = []
    provenance: Provenance

    def chains_for(self, accession: str) -> list[str]:
        return [c for e in self.entities if accession in e.uniprot_accessions for c in e.chains]


class AlphaFoldModel(BaseModel):
    model_entity_id: str
    accession: str
    gene: str | None = None
    organism: str | None = None
    sequence_start: int
    sequence_end: int
    model_version: int
    mean_plddt: float | None = None
    cif_url: str
    pdb_url: str | None = None
    pae_url: str | None = None
    plddt_url: str | None = None
    provenance: Provenance


class InterfaceResidue(BaseModel):
    chain: str
    residue_number: int
    insertion_code: str = ""
    residue_name: str
    partner_chain: str
    min_distance: float = Field(description="Closest heavy-atom distance to the partner chain (Å).")
    n_contacts: int = Field(description="Heavy-atom pairs within the cutoff.")


class Interface(BaseModel):
    pdb_id: str
    chain_a: str
    chain_b: str
    cutoff: float
    residues_a: list[InterfaceResidue]
    residues_b: list[InterfaceResidue]
    method: str = "Heavy-atom contacts between chains (gemmi ContactSearch), first model"
    provenance: Provenance

    @property
    def is_empty(self) -> bool:
        return not self.residues_a and not self.residues_b


class EnrichmentTerm(BaseModel):
    """One over-represented term from STRING functional enrichment."""

    category: str = Field(
        description="STRING category, e.g. KEGG, RCTM (Reactome), DISEASES, Process."
    )
    term: str
    description: str
    n_genes: int
    n_background: int
    p_value: float
    fdr: float
    genes: list[str]


class Enrichment(BaseModel):
    genes: list[str]
    species: int
    terms: list[EnrichmentTerm]
    provenance: Provenance

    def by_category(self, *categories: str) -> list[EnrichmentTerm]:
        return [t for t in self.terms if t.category in categories]


class DiseaseHit(BaseModel):
    disease_id: str
    name: str


class TargetAssociation(BaseModel):
    """An Open Targets disease–target association (overall score 0–1)."""

    ensembl_id: str
    symbol: str
    name: str | None = None
    uniprot: str | None = None
    score: float


class DiseaseAssociation(BaseModel):
    disease_id: str
    name: str
    score: float


class Pathway(BaseModel):
    pathway_id: str
    name: str
    top_level: str | None = None

    @property
    def url(self) -> str:
        return f"https://reactome.org/content/detail/{self.pathway_id}"


class DrugCandidate(BaseModel):
    chembl_id: str
    name: str | None = None
    drug_type: str | None = None
    max_stage: str
    mechanisms: list[str] = []
    targets: list[str] = []


class DiseaseProfile(BaseModel):
    disease_id: str
    name: str
    description: str | None = None
    therapeutic_areas: list[str] = []
    n_associated_targets: int
    targets: list[TargetAssociation]
    n_drug_candidates: int
    drugs: list[DrugCandidate]
    provenance: Provenance


class TargetProfile(BaseModel):
    ensembl_id: str
    symbol: str
    n_associated_diseases: int
    diseases: list[DiseaseAssociation]
    pathways: list[Pathway]
    provenance: Provenance
