"""Tests for the SemEval-2014 XML parser and loader (no network, no real data)."""

from __future__ import annotations

import pytest

from reviewlens.evaluation.semeval import OFFICIAL_FILES, load_semeval, parse_semeval_xml

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
  <sentence id="s1">
    <text>The fajitas were great but the service was awful.</text>
    <aspectTerms>
      <aspectTerm term="fajitas" polarity="positive" from="4" to="11"/>
      <aspectTerm term="service" polarity="negative" from="31" to="38"/>
    </aspectTerms>
  </sentence>
  <sentence id="s2">
    <text>Nothing aspect-worthy here.</text>
  </sentence>
  <sentence id="s3">
    <text>Huge &amp; varied wine list.</text>
    <aspectTerms>
      <aspectTerm term="wine list" polarity="conflict" from="14" to="23"/>
    </aspectTerms>
  </sentence>
</sentences>
"""


@pytest.fixture
def xml_path(tmp_path):
    path = tmp_path / "sample.xml"
    path.write_text(SAMPLE_XML, encoding="utf-8")
    return path


def test_parse_returns_all_sentences_including_termless(xml_path):
    sentences, terms = parse_semeval_xml(xml_path)
    assert len(sentences) == 3
    assert list(sentences.columns) == ["sentence_id", "text"]
    assert len(terms) == 3


def test_parse_extracts_term_fields_and_offsets(xml_path):
    _, terms = parse_semeval_xml(xml_path)
    fajitas = terms[terms["term"] == "fajitas"].iloc[0]
    assert fajitas["polarity"] == "positive"
    assert fajitas["start"] == 4 and fajitas["end"] == 11
    assert fajitas["sentence_id"] == "s1"


def test_parse_unescapes_entities(xml_path):
    sentences, _ = parse_semeval_xml(xml_path)
    s3 = sentences[sentences["sentence_id"] == "s3"].iloc[0]
    assert "&amp;" not in s3["text"] and "&" in s3["text"]


def test_parse_rejects_unknown_polarity(tmp_path):
    bad = SAMPLE_XML.replace('polarity="conflict"', 'polarity="meh"')
    path = tmp_path / "bad.xml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected polarity"):
        parse_semeval_xml(path)


def test_load_semeval_drops_conflict(tmp_path):
    # Write the sample under the official restaurants-test filename.
    path = tmp_path / OFFICIAL_FILES[("restaurants", "test")]
    path.write_text(SAMPLE_XML, encoding="utf-8")

    _, with_conflict = load_semeval("restaurants", "test", tmp_path, drop_conflict=False)
    _, without = load_semeval("restaurants", "test", tmp_path, drop_conflict=True)
    assert len(with_conflict) == 3
    assert len(without) == 2
    assert "conflict" not in set(without["polarity"])


def test_load_semeval_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_semeval"):
        load_semeval("laptops", "test", tmp_path)


def test_load_semeval_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset/split"):
        load_semeval("hotels", "test", tmp_path)
