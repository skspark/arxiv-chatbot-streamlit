# -*- coding:utf-8 -*-

from typing import List, Optional

from py_pdf_parser.components import PDFElement
from pydantic import BaseModel


class SectionTitle(BaseModel):
    elem_idx: int = -1
    font_size: int = 0
    content: str = ""
    is_abstract: bool = False
    is_overview: bool = False
    is_introduction: bool = False
    is_conclusion: bool = False
    is_references: bool = False


class PaperSection(BaseModel):
    title: Optional[SectionTitle] = None
    elems: List[PDFElement] = []

    class Config:
        arbitrary_types_allowed = True

    def summary(self):
        elems_summary = " ".join([item.text() for item in self.elems])
        return f"{self.title.content}: \n {elems_summary}"

    def chunked_elems_text(self, chunk_size: int) -> List[str]:
        if chunk_size <= 0:
            return []
        chunks = []
        c_chunk, c_chunk_size = [], 0
        for elem in self.elems:
            elem_txt = elem.text()
            elem_size = len(elem_txt)
            if c_chunk_size + len(elem_txt) >= chunk_size:
                if c_chunk:
                    chunks.append("".join(c_chunk))
                c_chunk, c_chunk_size = [elem_txt], elem_size
                continue
            c_chunk.append(elem_txt)
            c_chunk_size += elem_size
        if c_chunk:
            chunks.append("".join(c_chunk))
        return chunks


class PaperSections(BaseModel):
    abstract: Optional[PaperSection] = None
    overview: Optional[PaperSection] = None
    introduction: Optional[PaperSection] = None
    details: List[PaperSection] = []
    conclusion: Optional[PaperSection] = None
    references: Optional[PaperSection] = None

    def add(self, section: PaperSection):
        if section.title.is_abstract:
            self.abstract = section
        elif section.title.is_overview:
            self.overview = section
        elif section.title.is_introduction:
            self.introduction = section
        elif section.title.is_conclusion:
            self.conclusion = section
        elif section.title.is_references:
            self.references = section
        else:
            self.details.append(section)

    def summary(self) -> str:
        items = []
        if self.abstract:
            items.append(self.abstract.summary())
        if self.introduction:
            items.append(self.introduction.summary())
        if self.overview:
            items.append(self.overview.summary())
        for detail in self.details:
            items.append(detail.summary())
        if self.conclusion:
            items.append(self.conclusion.summary())
        if self.references:
            items.append(self.references.summary())
        return "\n\n".join(items)


class Paper(BaseModel):
    raw: str = ""
    title: List[str] = ""
    sections: Optional[PaperSections] = None
