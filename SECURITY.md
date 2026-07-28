# Security Policy

## Supported Versions

Security fixes are applied to the latest version on the default branch.

## Reporting a Vulnerability

Please report vulnerabilities privately and do not open public issues for active security problems.

Contact:

- `anton@kay.ai`

When possible, include:

- Affected component and version/commit
- Reproduction steps or proof of concept
- Impact assessment
- Suggested mitigation (if available)

We will acknowledge receipt and triage as quickly as possible.

## Disclosure

After a fix is available, we may publish a summary describing impact and remediation guidance.

## Untrusted Documents

Source documents are included in model prompts and must be treated as untrusted
input. Prompt-injection text can influence both count estimates and extracted
values. Count anomaly fallbacks reduce obvious failure modes but cannot detect
all plausible under-counts or fabricated output.

For adversarial inputs, use task-specific validation and an independent
coverage check, keep model credentials scoped, leave tracing disabled unless
appropriate, and do not execute or directly trust extracted content.
