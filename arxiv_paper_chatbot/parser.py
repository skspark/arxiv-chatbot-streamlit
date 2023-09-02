# -*- coding:utf-8 -*-

from typing import Optional

from arxiv_paper_chatbot.paper import Paper, PaperSection, PaperSections, SectionTitle
from py_pdf_parser.loaders import PDFDocument
from pydantic import BaseModel


class PaperParser(BaseModel):
    def parse(self, doc: PDFDocument) -> Optional[Paper]:
        if len(doc.pages) <= 0:
            return None
        paper = Paper()
        paper.title = self._parse_title(doc)
        paper.raw = " ".join([elem.text() for elem in doc.elements])
        paper.sections = self._get_sections(paper.title, doc)
        return paper

    """
    parse title from pdf document.
    detect title by retrieving largest text in first document.
    """

    @staticmethod
    def _parse_title(doc: PDFDocument) -> str:
        largest_elem = None
        first_page_elem_iter = iter(doc.pages[0].elements)
        while first_page_elem_iter:
            try:
                item = next(first_page_elem_iter)
                if not item.text() or len(item.text()) < 10:
                    continue
                largest_elem = (
                    item if not largest_elem else max(largest_elem, item, key=lambda a: a.font_size)
                )
            except StopIteration:
                break
        if largest_elem:
            return largest_elem.text()
        return ""

    """
    parse section titles from pdf document.
    """

    @staticmethod
    def _get_sections(paper_title: str, doc: PDFDocument) -> Optional[PaperSections]:
        orig_elems = list(doc.elements)
        if not orig_elems:
            return None
        elems_by_font_sizes = {}
        for elem_idx, elem in enumerate(doc.elements):
            elem_text = elem.text()
            if not elem_text or len(elem_text) < 1:
                continue
            elems_by_font_sizes.setdefault(elem.font_size, [])
            elems_by_font_sizes[elem.font_size].append((elem_idx, elem))

        # exclude elem group with less than 2 elem in group
        elems_by_font_sizes = {k: v for k, v in elems_by_font_sizes.items() if len(v) > 1}

        # exclude elem group that there font size is less than the ones who have most elements
        max_cnt_key = max(elems_by_font_sizes.keys(), key=lambda a: len(elems_by_font_sizes[a]))
        elems_by_font_sizes = {k: v for k, v in elems_by_font_sizes.items() if k > max_cnt_key}

        # check if any group contains the elem with "References" text.
        # if the elem group contains "References", it can be considered as section title elem group
        section_titles = []
        for font_size, elems in elems_by_font_sizes.items():
            if any("references" in item.text().lower() for _, item in elems):
                # parse section titles
                for elem_idx, item in elems:
                    item_text = item.text()
                    if len(item_text) < 5 or item_text == paper_title:
                        continue
                    item_text_lower = item_text.lower()
                    section_title = SectionTitle(
                        elem_idx=elem_idx,
                        font_size=font_size,
                        content=item_text,
                        is_abstract="abstract" in item_text_lower,
                        is_overview="overview" in item_text_lower,
                        is_introduction="intro" in item_text_lower
                        or "introduction" in item_text_lower,
                        is_conclusion="conclusion" in item_text_lower
                        or "result" in item_text_lower,
                        is_references="references" in item_text_lower,
                    )
                    section_titles.append(section_title)
        if not section_titles:
            return None

        # fill out the missing windows of the section, if section does not start from 0
        min_idx = min(item.elem_idx for item in section_titles)
        if min_idx != 0:
            start_section_title = SectionTitle(elem_idx=0, font_size=orig_elems[0].font_size)
            section_titles = [start_section_title] + section_titles

        sections = PaperSections()  # (section_title, section_elems)
        if len(section_titles) > 1:
            for i in range(len(section_titles) - 1):
                elems = orig_elems[(section_titles[i].elem_idx + 1) : section_titles[i + 1].elem_idx]
                section = PaperSection(
                    title=section_titles[i],
                    elems=[item.text() for item in elems],
                )
                sections.add(section)
        elems = orig_elems[section_titles[-1].elem_idx + 1 :]
        section = PaperSection(
            title=section_titles[-1],
            elems=[item.text() for item in elems],
        )
        sections.add(section)
        return sections
