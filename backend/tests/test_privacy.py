from app.services.privacy import scan_privacy


def test_privacy_scanner_masks_sensitive_values():
    findings = scan_privacy(
        "Contact editor@example.com or 617-555-0199. "
        "Never publish 123-45-6789."
    )

    assert [finding.kind for finding in findings] == ["email", "phone", "ssn"]
    assert findings[0].masked_value == "e***@example.com"
    assert "123-45-6789" not in findings[2].masked_value
    assert findings[2].risk == "high"


def test_privacy_scanner_does_not_return_raw_values():
    text = "Email jane.doe@example.org for details."

    finding = scan_privacy(text)[0]

    assert finding.masked_value != "jane.doe@example.org"
    assert finding.start == text.index("jane.doe@example.org")
