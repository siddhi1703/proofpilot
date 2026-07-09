from dataclasses import dataclass
import re

from app.schemas.audit import PrivacyFinding


@dataclass(frozen=True)
class PrivacyPattern:
    kind: str
    label: str
    pattern: re.Pattern[str]
    risk: str
    recommendation: str


PATTERNS = [
    PrivacyPattern(
        kind="email",
        label="Email address",
        pattern=re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
        risk="medium",
        recommendation="Confirm consent or replace the address with a role-based contact.",
    ),
    PrivacyPattern(
        kind="phone",
        label="Phone number",
        pattern=re.compile(
            r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
        ),
        risk="medium",
        recommendation="Remove or mask the number unless publication is authorized.",
    ),
    PrivacyPattern(
        kind="ssn",
        label="Possible U.S. Social Security number",
        pattern=re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        risk="high",
        recommendation="Do not publish. Remove the identifier and escalate for review.",
    ),
    PrivacyPattern(
        kind="credit_card",
        label="Possible payment-card number",
        pattern=re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"),
        risk="high",
        recommendation="Do not publish. Remove the number and follow incident procedures.",
    ),
    PrivacyPattern(
        kind="ip_address",
        label="IP address",
        pattern=re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
        ),
        risk="low",
        recommendation="Confirm that exposing this network identifier is necessary.",
    ),
]


def _mask(value: str) -> str:
    if "@" in value:
        local, domain = value.split("@", 1)
        return f"{local[:1]}***@{domain}"

    visible = [index for index, character in enumerate(value) if character.isalnum()]
    keep = set(visible[-4:])
    return "".join(
        character if index in keep or not character.isalnum() else "•"
        for index, character in enumerate(value)
    )


def scan_privacy(text: str) -> list[PrivacyFinding]:
    findings: list[PrivacyFinding] = []
    occupied_ranges: list[tuple[int, int]] = []

    for privacy_pattern in PATTERNS:
        for match in privacy_pattern.pattern.finditer(text):
            start, end = match.span()
            if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied_ranges):
                continue

            findings.append(
                PrivacyFinding(
                    kind=privacy_pattern.kind,
                    label=privacy_pattern.label,
                    masked_value=_mask(match.group()),
                    risk=privacy_pattern.risk,
                    start=start,
                    end=end,
                    recommendation=privacy_pattern.recommendation,
                )
            )
            occupied_ranges.append((start, end))

    return sorted(findings, key=lambda finding: finding.start)
