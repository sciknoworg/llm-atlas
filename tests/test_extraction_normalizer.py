"""Tests for extraction normalizer date handling."""

from src.extraction_normalizer import (
    format_published_date_from_metadata,
    normalize_date_created,
    normalize_extraction,
)


class TestPublishedDateFromMetadata:
    def test_iso_datetime_with_timezone(self):
        assert (
            format_published_date_from_metadata("2023-06-14T18:00:00+00:00")
            == "2023-06-14"
        )

    def test_date_only(self):
        assert format_published_date_from_metadata("2023-06-14") == "2023-06-14"

    def test_zulu_suffix(self):
        assert format_published_date_from_metadata("2023-06-14T18:00:00Z") == "2023-06-14"


class TestNormalizeDateCreated:
    def test_preserves_full_date(self):
        assert normalize_date_created("2018-10-01") == "2018-10-01"

    def test_preserves_month_only(self):
        assert normalize_date_created("2018-10") == "2018-10"

    def test_year_expands_to_first_of_year(self):
        assert normalize_date_created("2018") == "2018-01-01"

    def test_normalize_extraction_applies_date(self):
        models = [{"model_name": "X", "date_created": "2018-10-01"}]
        out = normalize_extraction(models)
        assert out[0]["date_created"] == "2018-10-01"
